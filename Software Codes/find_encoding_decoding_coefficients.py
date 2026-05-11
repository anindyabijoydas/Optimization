# -*- coding: utf-8 -*-
"""
Coded distributed computing design optimizer — H100-optimized.

No-ridge residual, double-precision variant.

This version computes each subset residual from the unregularized normal
equations, solving gram @ r = rhs instead of
(gram + ridge * I) @ r = rhs.

The condition penalty still uses a small diagonal stabilizer so the Cholesky
factorization used for trace((gram + condition_ridge * I)^-1) remains well
defined.

Speed-critical changes vs previous version:
  - Replaced eigvalsh (batched cuSOLVER, serialized per-matrix) with
    trace(stabilized^-1) as the condition penalty.
    trace(A^-1) = ||L^-1||^2_F where L is the Cholesky factor, computed via
    a single triangular solve against the identity — no eigendecomposition.
    This is mathematically valid: trace(A^-1) >= 1/lambda_min(A), so it is a
    tight upper-bound surrogate for 1/lambda_min that penalizes ill-conditioning
    even more aggressively, with correct gradients and far lower GPU cost.
  - torch.compile("default") retained with the pad_mm patch.
  - eval_obj closed over all non-tensor constants (no graph breaks).
  - All 9880 subsets processed in one shot on CUDA (no Python loop overhead).

Converted from Jupyter notebook for use in Spyder (Anaconda).
Run with F5, or run individual cells with Ctrl+Enter.
"""

# %% Imports and global configuration

from pathlib import Path
from itertools import combinations
from time import perf_counter

import torch
import random
from scipy.io import savemat

torch.set_default_dtype(torch.float64)
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# ── Problem dimensions ────────────────────────────────────────────────────────
n_workers   = 7
dim         = 3
subset_size = 5   

alpha         = torch.eye(dim, dtype=torch.get_default_dtype()).reshape(-1)
subsets       = list(combinations(range(n_workers), subset_size))
subset_matrix = torch.tensor(subsets, dtype=torch.long)   # (9880, 37)

condition_ridge, eps          = 1e-6, 1e-6
lam_resid, lam_cond, lam_norm = 1.0, 1e-6, 1e-8

# ── Runtime config ────────────────────────────────────────────────────────────
cuda_subset_batch_size = len(subsets)   # one-shot: no Python loop on GPU path
cpu_subset_batch_size  = 256
enable_cuda_compile    = False  # torch.compile not supported on Windows
cuda_compile_mode      = "default"

# ── Patch the pad_mm inductor bug ─────────────────────────────────────────────
# should_pad_bench() calls torch.randn() at compile time → crashes.
# Returning False skips the padding heuristic with no effect on results.
try:
    import torch._inductor.fx_passes.pad_mm as _pad_mm
    _pad_mm.should_pad_bench = lambda *a, **kw: False
except Exception:
    pass

_runtime_cache  = {}
_eval_obj_cache = {}


# %% Config / tensor cache

def get_runtime_config(device=None):
    if device == "cuda":
        return {"device": "cuda", "dtype": torch.float64,
                "subset_batch_size": cuda_subset_batch_size}
    return {"device": device, "dtype": torch.float64,
            "subset_batch_size": cpu_subset_batch_size}


def get_runtime_tensors(device, dtype):
    cache_key = (str(device), dtype)
    if cache_key not in _runtime_cache:
        _runtime_cache[cache_key] = {
            "alpha_norm_sq": alpha.square().sum().to(device=device, dtype=dtype),
            "subset_matrix": subset_matrix.to(device=device),
            "eye_subset":    torch.eye(subset_size, dtype=dtype, device=device),
        }
    return _runtime_cache[cache_key]


# %% Math helpers

def compute_gram_quantities(X, Y):
    """
    gram_workers  (n, n)  — Hadamard product of X Xt and Y Yt
    rhs_workers   (n,)    — row-wise dot products of X and Y
    worker_norms  (n,)    — sqrt of gram diagonal
    """
    xx           = X @ X.T
    yy           = Y @ Y.T
    gram_workers = xx * yy
    rhs_workers  = (X * Y).sum(dim=1)
    worker_norms = torch.sqrt(torch.clamp(gram_workers.diagonal(), min=0.0))
    return gram_workers, rhs_workers, worker_norms


def batched_subset_gram(gram_workers, subset_chunk):
    """(B, s, s) subset Gram matrices gathered from the full (n, n) matrix."""
    return gram_workers[subset_chunk.unsqueeze(-1), subset_chunk.unsqueeze(-2)]


def batched_subset_rhs(rhs_workers, subset_chunk):
    """(B, s, 1) RHS vectors."""
    return rhs_workers[subset_chunk].unsqueeze(-1)


# %% Objective function factory
# All non-tensor constants are closed over so torch.compile sees a pure
# (X, Y) -> scalar function with no Python-scalar arguments → no graph breaks.
#
# Condition penalty: trace(stabilized^{-1}) instead of 1/lambda_min
# -----------------------------------------------------------------------
# For a PD matrix A with Cholesky factor L (A = L Lᵀ):
#
#   trace(A^{-1}) = ||L^{-1}||^2_F
#
# L^{-1} is obtained cheaply via triangular_solve(I, L, upper=False).
# This is O(s²) per matrix vs O(s³) for eigvalsh, with much better GPU
# utilization because triangular_solve is a single batched BLAS-3 kernel.
#
# As a penalty, trace(A^{-1}) >= 1/lambda_min(A) with equality only when
# A is a multiple of the identity, so it is a valid (tighter) surrogate
# for ill-conditioning that correctly drives lambda_min upward.

def _make_eval_obj(subset_matrix_device, eye_subset, alpha_norm_sq, batch_size):
    def eval_obj(X, Y):
        gram_workers, rhs_workers, worker_norms = compute_gram_quantities(X, Y)
        dtype  = gram_workers.dtype
        device = gram_workers.device

        # eye broadcast for the triangular solve  (1, s, s)
        eye_b = eye_subset.unsqueeze(0)

        total = torch.zeros(1, dtype=dtype, device=device)

        for start in range(0, subset_matrix_device.shape[0], batch_size):
            subset_chunk = subset_matrix_device[start : start + batch_size]
            gram = batched_subset_gram(gram_workers, subset_chunk)   # (B, s, s)
            rhs  = batched_subset_rhs(rhs_workers, subset_chunk)     # (B, s, 1)

            resid_chol = torch.linalg.cholesky(gram + 1e-10 * eye_b)                 # (B, s, s)

            # ── Decode ───────────────────────────────────────────────────────
            r = torch.cholesky_solve(rhs, resid_chol)                 # (B, s, 1)

            linear_term = (rhs.transpose(1, 2) @ r).squeeze(-1).squeeze(-1)
            resid_sq    = alpha_norm_sq - linear_term

            stabilized = gram + condition_ridge * eye_b              # (B, s, s)
            chol       = torch.linalg.cholesky(stabilized)           # (B, s, s)

            # ── Condition penalty via trace(stabilized^{-1}) ─────────────────
            # Solve L @ Linv = I  →  Linv = L^{-1}  (lower-triangular)
            # ||Linv||^2_F = trace(A^{-1})
            Linv        = torch.linalg.solve_triangular(
                              chol, eye_b.expand(chol.shape[0], -1, -1),
                              upper=False)                            # (B, s, s)
            trace_inv   = Linv.pow(2).sum(dim=(-2, -1))              # (B,)

            total = total + (lam_resid * resid_sq + lam_cond * trace_inv).sum()

        total = total + lam_norm * (worker_norms - 1.0).pow(2).sum()
        return total.squeeze(0)

    return eval_obj


def get_eval_obj_fn(device, dtype, subset_batch_size):
    cache_key = (str(device), dtype, subset_batch_size)
    if cache_key in _eval_obj_cache:
        return _eval_obj_cache[cache_key]

    tensors  = get_runtime_tensors(device, dtype)
    eval_obj = _make_eval_obj(
        tensors["subset_matrix"],
        tensors["eye_subset"],
        tensors["alpha_norm_sq"],
        subset_batch_size,
    )

    if str(device) == "cuda" and enable_cuda_compile and hasattr(torch, "compile"):
        fn = torch.compile(eval_obj, mode=cuda_compile_mode)
    else:
        fn = eval_obj

    _eval_obj_cache[cache_key] = fn
    return fn


# %% Initialisation

def init_factors(seed=0, device=None):
    rc = get_runtime_config(device)
    generator = torch.Generator(device=rc["device"])
    generator.manual_seed(seed)
    X = torch.randn((n_workers, dim), generator=generator,
                    device=rc["device"], dtype=rc["dtype"], requires_grad=True)
    Y = torch.randn((n_workers, dim), generator=generator,
                    device=rc["device"], dtype=rc["dtype"], requires_grad=True)
    return X, Y


# %% Optimisation loop

def optimize_design(
    num_steps=20000, lr=1e-2, seed=0, device=None,
    verbose=True, subset_batch_size=None,
    lr_patience=200, lr_factor=0.5, min_lr=1e-7,
    early_stop_patience=800, target_tolerance=1e-10,
):
    rc = get_runtime_config(device)
    if subset_batch_size is None:
        subset_batch_size = rc["subset_batch_size"]

    eval_obj_fn = get_eval_obj_fn(rc["device"], rc["dtype"], subset_batch_size)
    X, Y        = init_factors(seed=seed, device=device)
    optimizer   = torch.optim.Adam([X, Y], lr=lr)
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_factor,
            patience=lr_patience,
            threshold=1e-8,
            threshold_mode="rel",
            min_lr=min_lr,
            verbose=True,
        )
    except TypeError:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_factor,
            patience=lr_patience,
            threshold=1e-8,
            threshold_mode="rel",
            min_lr=min_lr,
        )
    history     = []
    step_times  = []
    is_cuda     = (device == "cuda")
    best_loss   = float("inf")
    best_step   = -1
    best_X      = None
    best_Y      = None
    steps_since_best = 0

    for step in range(num_steps):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = perf_counter()

        optimizer.zero_grad()
        loss = eval_obj_fn(X, Y)
        loss.backward()
        optimizer.step()

        if is_cuda:
            torch.cuda.synchronize()
        step_time = perf_counter() - t0

        current_loss = loss.item()
        scheduler.step(current_loss)

        if current_loss < best_loss:
            best_loss = current_loss
            best_step = step
            best_X = X.detach().clone()
            best_Y = Y.detach().clone()
            steps_since_best = 0
        else:
            steps_since_best += 1

        history.append(current_loss)
        step_times.append(step_time)

        if verbose and (step == 0 or (step + 1) % 1000 == 0):
            avg = sum(step_times) / len(step_times)
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"step={step+1:4d}, objective={current_loss:.6e}, "
                  f"best_objective={best_loss:.6e}, lr={current_lr:.3e}, "
                  f"step_time={step_time:.3f}s, avg_step_time={avg:.3f}s")

        if best_loss < target_tolerance:
            if verbose:
                print(f"\nConverged at step {step + 1}.")
            break

        if steps_since_best >= early_stop_patience:
            if verbose:
                print(f"\nEarly stopping at step {step + 1}: no improvement.")
            break

    if verbose:
        print(f"Best step: {best_step + 1}")
        print(f"Best objective: {best_loss:.15e}")

    return best_X, best_Y, history, step_times


def optimize_best_of_seeds(
    seeds, num_steps=20000, lr=1e-2, device=None,
    verbose=True, subset_batch_size=None,
):
    rc = get_runtime_config(device)
    if subset_batch_size is None:
        subset_batch_size = rc["subset_batch_size"]

    best_result = None
    for seed in seeds:
        if verbose:
            print(f"\n=== Seed {seed} ===")
        X, Y, history, step_times = optimize_design(
            num_steps=num_steps, lr=lr, seed=seed, device=device,
            verbose=verbose, subset_batch_size=subset_batch_size,
        )
        final_objective = history[-1]
        if verbose:
            print(f"seed={seed:2d}, final objective={final_objective:.6e}")
        candidate = {"seed": seed, "X": X, "Y": Y, "history": history,
                     "step_times": step_times, "final_objective": final_objective}
        if best_result is None or final_objective < best_result["final_objective"]:
            best_result = candidate
    return best_result


# %% Post-processing  (correctness > speed; eigvalsh is fine here)

@torch.no_grad()
def compute_decoding_coefficients(X, Y, subset_batch_size=None):
    if subset_batch_size is None:
        subset_batch_size = get_runtime_config(X.device.type)["subset_batch_size"]
    gram_workers, rhs_workers, _ = compute_gram_quantities(X, Y)
    tensors = get_runtime_tensors(gram_workers.device, gram_workers.dtype)
    SM  = tensors["subset_matrix"]
    chunks = []
    for start in range(0, SM.shape[0], subset_batch_size):
        sc         = SM[start : start + subset_batch_size]
        gram       = batched_subset_gram(gram_workers, sc)
        rhs        = batched_subset_rhs(rhs_workers, sc)
        chol       = torch.linalg.cholesky(gram)
        r          = torch.cholesky_solve(rhs, chol).squeeze(-1)
        chunks.append(r)
    return torch.cat(chunks, dim=0)   # (num_subsets, subset_size)


@torch.no_grad()
def compute_condition_numbers(X, Y, subset_batch_size=None):
    """True condition numbers via eigvalsh — called only at save time."""
    if subset_batch_size is None:
        subset_batch_size = get_runtime_config(X.device.type)["subset_batch_size"]
    gram_workers, _, _ = compute_gram_quantities(X, Y)
    tensors = get_runtime_tensors(gram_workers.device, gram_workers.dtype)
    SM  = tensors["subset_matrix"]
    eye = tensors["eye_subset"]
    chunks = []
    for start in range(0, SM.shape[0], subset_batch_size):
        sc         = SM[start : start + subset_batch_size]
        gram       = batched_subset_gram(gram_workers, sc)
        stabilized = gram + condition_ridge * eye.unsqueeze(0)
        eigvals    = torch.linalg.eigvalsh(stabilized)          # ascending
        sigma_min  = torch.sqrt(torch.clamp(eigvals[:,  0], min=0.0))
        sigma_max  = torch.sqrt(torch.clamp(eigvals[:, -1], min=0.0))
        chunks.append(sigma_max / torch.clamp(sigma_min, min=1e-15))
    cond_tensor = torch.cat(chunks, dim=0)
    return cond_tensor, cond_tensor.max().item(), cond_tensor.mean().item()


# %% MATLAB export

def save_to_mat(X, Y, history, step_times=None, best_seed=None,
                output_path=None, subset_batch_size=None):

    if output_path is None:
        output_path = Path(
            f"coded_design_{subset_size}_of_{n_workers}_torch_batched_result_{lam_cond}_{lam_norm}.mat"
        )
    else:
        output_path = Path(output_path)


    if subset_batch_size is None:
        subset_batch_size = get_runtime_config(X.device.type)["subset_batch_size"]

    decoding_coeff = compute_decoding_coefficients(X, Y, subset_batch_size=subset_batch_size)
    cond_values, worst_cond, avg_cond = compute_condition_numbers(
        X, Y, subset_batch_size=subset_batch_size)

    savemat(output_path, {
        "X": X.cpu().numpy(), "Y": Y.cpu().numpy(),
        "history": history,
        "step_times": [] if step_times is None else step_times,
        "best_seed": -1 if best_seed is None else best_seed,
        "alpha": alpha.cpu().numpy(),
        "subset_matrix": subset_matrix.cpu().numpy(),
        "decoding_coeff": decoding_coeff.cpu().numpy(),
        "condition_numbers": cond_values.cpu().numpy(),
        "worst_condition_number": worst_cond,
        "average_condition_number": avg_cond,
        "subset_batch_size": subset_batch_size,
        "condition_ridge": condition_ridge, "eps": eps,
        "lam_resid": lam_resid, "lam_cond": lam_cond, "lam_norm": lam_norm,
    })
    return output_path, worst_cond, avg_cond


# %% Entry point

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else None
    rc     = get_runtime_config(device)

    if device == "cuda":
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        print(f"CUDA runtime config: dtype={rc['dtype']}, "
              f"subset_batch_size={rc['subset_batch_size']}, "
              f"compile={enable_cuda_compile}, compile_mode={cuda_compile_mode}")

    # Step 1 is slow (torch.compile traces). Steady-state from step 2 onward.
    tmp = random.sample(range(1, 101), 20)
    best_result = optimize_best_of_seeds(
        seeds=tmp, num_steps=20000, lr=1e-2,
        device=device, verbose=True,
        subset_batch_size=rc["subset_batch_size"],
    )

    X, Y       = best_result["X"], best_result["Y"]
    history    = best_result["history"]
    step_times = best_result["step_times"]
    best_seed  = best_result["seed"]

    mat_path, worst_cond, avg_cond = save_to_mat(
        X, Y, history, step_times=step_times, best_seed=best_seed,
        subset_batch_size=rc["subset_batch_size"],
    )

    print("\nBest seed:", best_seed)
    print("Final objective:", history[-1])
    print(f"Average step time: {sum(step_times)/len(step_times):.3f}s")
    print(f"Worst-case condition number: {worst_cond:.6e}")
    print(f"Average condition number:    {avg_cond:.6e}")
    print("Saved MATLAB file:", mat_path)
