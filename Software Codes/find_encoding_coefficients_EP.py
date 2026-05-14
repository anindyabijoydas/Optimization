# -*- coding: utf-8 -*-
"""
Created on Wed May  6 19:19:56 2026

@author: adas
"""

# %% Imports and global configuration

from pathlib import Path
from itertools import combinations
from time import perf_counter
import random

import torch
from scipy.io import savemat, loadmat

torch.set_default_dtype(torch.float64)
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# ── Problem dimensions ────────────────────────────────────────────────────────
# Set the four parameters of the problem here.
N      = 11      # total number of worker nodes
m      = 2       # column-blocks of A
n      = 2       # column-blocks of B
p      = 2       # shared row-blocks of A and B
thresh = p * m * n + p - 1   # decoding threshold (= 9 for m=n=p=2)
                              # set to a different value if you want; the
                              # formula above is the standard "rank" bound.

s_stragglers = N - thresh    # tolerated stragglers (informational)

# Number of decoding targets (the m·n α-matrices α_{j,k}).
num_targets = m * n

# Frobenius-norm-squared of every target α_{j,k} = e_j e_k^T ⊗ I_p is p.
target_norm_sq = float(p)

# Build subset table: every size-`thresh` subset of {0,…,N-1}.
subsets       = list(combinations(range(N), thresh))
subset_matrix = torch.tensor(subsets, dtype=torch.long)        # (num_subsets, thresh)
num_subsets   = subset_matrix.shape[0]

condition_ridge, eps          = 1e-4, 1e-6
lam_resid, lam_cond, lam_norm = 1.0, 1e-4, 1e-8

# ── Run config ────────────────────────────────────────────────────────────────
# Multi-seed sweep: pick NUM_SEEDS distinct seeds at random from [1..SEED_POOL],
# run the full optimization for each, keep the best.
NUM_SEEDS    = 5        # how many random seeds to try
SEED_POOL    = 100       # seeds drawn uniformly from {1, 2, ..., SEED_POOL}
META_SEED    = 0         # seed used for the random.sample() call itself, so
                         #   the chosen seed list is reproducible across runs.
                         #   Set to None for a fresh draw every time.

NUM_STEPS    = 20000
LEARN_RATE   = 1e-2

# Initialisation persistence
#   With multi-seed, USE_INIT/SAVE_INIT only make sense if you want to lock the
#   sweep to a single specific starting point — i.e. NUM_SEEDS=1 and the chosen
#   seed becomes irrelevant because the init is loaded from disk. The hooks are
#   left in place for parity with the single-seed file but default OFF.
SAVE_INIT  = 0
USE_INIT   = 0

# ── Runtime config ────────────────────────────────────────────────────────────
cuda_subset_batch_size = num_subsets   # one-shot on GPU when feasible
cpu_subset_batch_size  = 256
enable_cuda_compile    = False         # torch.compile not supported on Windows
cuda_compile_mode      = "default"

# Patch the pad_mm inductor bug (no-op outside torch.compile)
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
            "subset_matrix":  subset_matrix.to(device=device),
            "eye_thresh":     torch.eye(thresh, dtype=dtype, device=device),
            "target_norm_sq": torch.tensor(target_norm_sq,
                                            dtype=dtype, device=device),
        }
    return _runtime_cache[cache_key]


# %% Math helpers

def compute_gram_quantities(X, Y):
    """
    Inputs
    ------
    X : (N, p, m) — encoding of A-blocks (one matrix per worker)
    Y : (N, p, n) — encoding of B-blocks (one matrix per worker)

    Returns
    -------
    gram_workers     : (N, N)        ⟨X_a,X_b⟩_F · ⟨Y_a,Y_b⟩_F
    rhs_workers_flat : (N, m·n)      reshaped from (N, m, n) with
                                     R[ell, j, k] = (X_ell^T Y_ell)[j, k]
    worker_norms     : (N,)          sqrt of the gram diagonal
    """
    xx_inner     = torch.einsum("aij,bij->ab", X, X)        # (N, N)
    yy_inner     = torch.einsum("aij,bij->ab", Y, Y)        # (N, N)
    gram_workers = xx_inner * yy_inner                      # Hadamard

    # Per-worker target inner products: (N, m, n)
    rhs_workers      = torch.einsum("aij,aik->ajk", X, Y)
    rhs_workers_flat = rhs_workers.reshape(X.shape[0], m * n)

    worker_norms = torch.sqrt(torch.clamp(gram_workers.diagonal(), min=0.0))
    return gram_workers, rhs_workers_flat, worker_norms


def batched_subset_gram(gram_workers, subset_chunk):
    """(B, thresh, thresh) gathered submatrices."""
    return gram_workers[subset_chunk.unsqueeze(-1), subset_chunk.unsqueeze(-2)]


def batched_subset_rhs(rhs_workers_flat, subset_chunk):
    """(B, thresh, m·n) — one column per α_{j,k} target."""
    return rhs_workers_flat[subset_chunk]


# %% Objective function factory
#
# Per-subset residual (sum over m·n targets):
#     resid_S = m·n·p − Σ_{j,k} r^{(j,k),T}_S G_S^{-1} r^{(j,k)}_S
#
# Equivalently, with R the (thresh × m·n) RHS stack and L = chol(G_S):
#     resid_S = m·n·p − ||L^{-1} R||_F^2
#
# Condition penalty: trace((G_S + ridge·I)^{-1}) = ||L^{-1}||_F^2 with the
# *stabilized* gram. Same as in the original code.

def _make_eval_obj(subset_matrix_device, eye_thresh, target_norm_sq, batch_size):
    def eval_obj(X, Y):
        gram_workers, rhs_workers_flat, worker_norms = compute_gram_quantities(X, Y)
        dtype  = gram_workers.dtype
        device = gram_workers.device

        eye_b = eye_thresh.unsqueeze(0)                     # (1, thresh, thresh)
        total = torch.zeros(1, dtype=dtype, device=device)

        for start in range(0, subset_matrix_device.shape[0], batch_size):
            subset_chunk = subset_matrix_device[start : start + batch_size]
            gram = batched_subset_gram(gram_workers, subset_chunk)        # (B, t, t)
            rhs  = batched_subset_rhs(rhs_workers_flat, subset_chunk)     # (B, t, mn)

            # ── Decode (least-squares for each of the m·n targets) ───────────
            resid_chol = torch.linalg.cholesky(gram)                       # (B, t, t)
            r          = torch.cholesky_solve(rhs, resid_chol)             # (B, t, mn)

            # Σ_{j,k} r^T_{j,k} (G^{-1} r_{j,k}) for each subset → (B,)
            linear_term_per_target = (rhs * r).sum(dim=1)                  # (B, mn)
            resid_sq = (target_norm_sq * (m * n)
                        - linear_term_per_target.sum(dim=-1))              # (B,)

            # ── Condition penalty trace((G + ridge·I)^{-1}) ─────────────────
            stabilized = gram + condition_ridge * eye_b
            chol       = torch.linalg.cholesky(stabilized)
            Linv       = torch.linalg.solve_triangular(
                             chol, eye_b.expand(chol.shape[0], -1, -1),
                             upper=False)
            trace_inv  = Linv.pow(2).sum(dim=(-2, -1))                     # (B,)

            total = total + (lam_resid * resid_sq + lam_cond * trace_inv).sum()

        # Norm penalty: keep ||flatX_ell|| · ||flatY_ell|| ≈ 1 (same as original)
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
        tensors["eye_thresh"],
        tensors["target_norm_sq"],
        subset_batch_size,
    )

    if str(device) == "cuda" and enable_cuda_compile and hasattr(torch, "compile"):
        fn = torch.compile(eval_obj, mode=cuda_compile_mode)
    else:
        fn = eval_obj

    _eval_obj_cache[cache_key] = fn
    return fn


# %% Initialisation

def _init_file_path():
    """File name encodes the full (m, n, p, N, thresh) signature."""
    return Path(f"init_m{m}_n{n}_p{p}_N{N}_thresh{thresh}.mat")


def init_factors(seed=0, device=None):
    rc = get_runtime_config(device)
    init_path = _init_file_path()

    # ── Load mode ────────────────────────────────────────────────────────────
    if USE_INIT:
        if not init_path.exists():
            raise FileNotFoundError(
                f"USE_INIT=1 but no init file found at: {init_path.resolve()}\n"
                f"Run once with SAVE_INIT=1 to create it."
            )
        data = loadmat(init_path)
        X_np, Y_np = data["X_init"], data["Y_init"]
        if X_np.shape != (N, p, m) or Y_np.shape != (N, p, n):
            raise ValueError(
                f"Init file shape mismatch: expected X={ (N, p, m) }, "
                f"Y={ (N, p, n) }; got X={X_np.shape}, Y={Y_np.shape}"
            )
        X = torch.tensor(X_np, dtype=rc["dtype"],
                         device=rc["device"]).requires_grad_(True)
        Y = torch.tensor(Y_np, dtype=rc["dtype"],
                         device=rc["device"]).requires_grad_(True)
        print(f"Loaded init from: {init_path.resolve()}")
        return X, Y

    # ── Fresh random init ────────────────────────────────────────────────────
    generator = torch.Generator(device=rc["device"])
    generator.manual_seed(seed)
    X = torch.randn((N, p, m), generator=generator,
                    device=rc["device"], dtype=rc["dtype"], requires_grad=True)
    Y = torch.randn((N, p, n), generator=generator,
                    device=rc["device"], dtype=rc["dtype"], requires_grad=True)

    if SAVE_INIT:
        savemat(init_path, {
            "X_init":      X.detach().cpu().numpy(),
            "Y_init":      Y.detach().cpu().numpy(),
            "seed":        seed,
            "N":           N,
            "m":           m,
            "n":           n,
            "p":           p,
            "thresh":      thresh,
        })
        print(f"Saved init to: {init_path.resolve()}")

    return X, Y


# %% Optimisation loop (single seed)

def optimize_design(
    num_steps=40000, lr=1e-2, seed=0, device=None,
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
            optimizer, mode="min", factor=lr_factor, patience=lr_patience,
            threshold=1e-8, threshold_mode="rel", min_lr=min_lr, verbose=True,
        )
    except TypeError:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=lr_factor, patience=lr_patience,
            threshold=1e-8, threshold_mode="rel", min_lr=min_lr,
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
            print(f"step={step+1:5d}, objective={current_loss:.6e}, "
                  f"best_objective={best_loss:.6e}, lr={current_lr:.3e}, "
                  f"avg_step_time={avg:.3f}s")

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
    """
    Run optimize_design once per seed and return the result with the lowest
    final objective (the value at the last step that ran — same semantics as
    the original m=n=1 multi-seed code).

    Returns a dict containing the best run's encodings, optimization history,
    step timings, seed, final objective, and a per-seed summary list.
    """
    rc = get_runtime_config(device)
    if subset_batch_size is None:
        subset_batch_size = rc["subset_batch_size"]

    summary     = []        # [(seed, final_objective, best_objective_seen), ...]
    best_result = None

    for seed in seeds:
        if verbose:
            print(f"\n=== Seed {seed} ===")
        X, Y, history, step_times = optimize_design(
            num_steps=num_steps, lr=lr, seed=seed, device=device,
            verbose=verbose, subset_batch_size=subset_batch_size,
        )
        final_objective = history[-1]
        best_objective  = min(history)
        if verbose:
            print(f"seed={seed:3d}  final={final_objective:.6e}  "
                  f"best_seen={best_objective:.6e}")
        summary.append((seed, final_objective, best_objective))

        candidate = {
            "seed":            seed,
            "X":               X,
            "Y":               Y,
            "history":         history,
            "step_times":      step_times,
            "final_objective": final_objective,
            "best_objective":  best_objective,
        }
        if (best_result is None
                or final_objective < best_result["final_objective"]):
            best_result = candidate

    best_result["summary"] = summary
    return best_result


# %% Post-processing

@torch.no_grad()
def compute_decoding_coefficients(X, Y, subset_batch_size=None):
    """
    For each subset S and each target (j,k), compute
        c^{(j,k)}_S = G_S^{-1} r^{(j,k)}_S    ∈ R^{thresh}.
    Return shape: (num_subsets, thresh, m, n).

    Decoding: (A^T B)_{j,k} = Σ_{ℓ ∈ S} c^{(j,k)}_S[ℓ-position] · C̃_ℓ
    where C̃_ℓ = Ã_ℓ^T B̃_ℓ is the product computed by worker ℓ.
    """
    if subset_batch_size is None:
        subset_batch_size = get_runtime_config(X.device.type)["subset_batch_size"]
    gram_workers, rhs_workers_flat, _ = compute_gram_quantities(X, Y)
    tensors = get_runtime_tensors(gram_workers.device, gram_workers.dtype)
    SM      = tensors["subset_matrix"]
    chunks  = []
    for start in range(0, SM.shape[0], subset_batch_size):
        sc   = SM[start : start + subset_batch_size]
        gram = batched_subset_gram(gram_workers, sc)
        rhs  = batched_subset_rhs(rhs_workers_flat, sc)              # (B, t, mn)
        chol = torch.linalg.cholesky(gram)
        c    = torch.cholesky_solve(rhs, chol)                       # (B, t, mn)
        chunks.append(c)
    coeff_flat = torch.cat(chunks, dim=0)                            # (num, t, mn)
    return coeff_flat.reshape(SM.shape[0], thresh, m, n)


@torch.no_grad()
def compute_condition_numbers(X, Y, subset_batch_size=None):
    """True condition numbers via eigvalsh — called only at save time."""
    if subset_batch_size is None:
        subset_batch_size = get_runtime_config(X.device.type)["subset_batch_size"]
    gram_workers, _, _ = compute_gram_quantities(X, Y)
    tensors = get_runtime_tensors(gram_workers.device, gram_workers.dtype)
    SM  = tensors["subset_matrix"]
    eye = tensors["eye_thresh"]
    chunks = []
    for start in range(0, SM.shape[0], subset_batch_size):
        sc         = SM[start : start + subset_batch_size]
        gram       = batched_subset_gram(gram_workers, sc)
        stabilized = gram + condition_ridge * eye.unsqueeze(0)
        eigvals    = torch.linalg.eigvalsh(stabilized)
        sigma_min  = torch.sqrt(torch.clamp(eigvals[:,  0], min=0.0))
        sigma_max  = torch.sqrt(torch.clamp(eigvals[:, -1], min=0.0))
        chunks.append(sigma_max / torch.clamp(sigma_min, min=1e-15))
    cond_tensor = torch.cat(chunks, dim=0)
    return cond_tensor, cond_tensor.max().item(), cond_tensor.mean().item()


@torch.no_grad()
def build_target_matrices():
    """
    Construct the m·n target matrices α_{j,k} = e_j e_k^T ⊗ I_p explicitly,
    purely for archival inspection in the .mat file. Shape: (m, n, p·m, p·n).
    Not used in the optimization (the math only needs ⟨V_ℓ, α_{j,k}⟩_F).
    """
    targets = torch.zeros(m, n, p * m, p * n)
    for j in range(m):
        for k in range(n):
            for i in range(p):
                row = j * p + i
                col = k * p + i
                targets[j, k, row, col] = 1.0
    return targets


# %% MATLAB export

def save_to_mat(X, Y, history, step_times=None, best_seed=None,
                seed_summary=None, output_path=None, subset_batch_size=None):

    if output_path is None:
        output_path = Path(
            f"coded_design_m{m}_n{n}_p{p}_N{N}_thresh{thresh}"
            f"_torch_batched_result.mat"
        )
    else:
        output_path = Path(output_path)

    if subset_batch_size is None:
        subset_batch_size = get_runtime_config(X.device.type)["subset_batch_size"]

    decoding_coeff = compute_decoding_coefficients(
        X, Y, subset_batch_size=subset_batch_size)            # (num, t, m, n)
    cond_values, worst_cond, avg_cond = compute_condition_numbers(
        X, Y, subset_batch_size=subset_batch_size)

    targets = build_target_matrices()

    # Convert seed_summary [(seed, final, best_seen), ...] to a numeric array
    # so MATLAB can read it as a (NUM_SEEDS x 3) double array.
    if seed_summary is None or len(seed_summary) == 0:
        seed_summary_array = []
    else:
        seed_summary_array = [[float(s), float(f), float(b)]
                              for (s, f, b) in seed_summary]

    savemat(output_path, {
        # Encodings
        "X": X.cpu().numpy(),                  # (N, p, m)
        "Y": Y.cpu().numpy(),                  # (N, p, n)
        # Optimization trace (best seed only)
        "history":           history,
        "step_times":        [] if step_times is None else step_times,
        "best_seed":         -1 if best_seed is None else best_seed,
        "seed_summary":      seed_summary_array,   # rows: [seed, final, best_seen]
        # Problem parameters
        "N":                 N,
        "m":                 m,
        "n":                 n,
        "p":                 p,
        "thresh":            thresh,
        "s_stragglers":      s_stragglers,
        "subset_matrix":     subset_matrix.cpu().numpy(),     # (num_subsets, t)
        # Targets and decoding
        "targets":           targets.cpu().numpy(),           # (m, n, p*m, p*n)
        "decoding_coeff":    decoding_coeff.cpu().numpy(),    # (num, t, m, n)
        "condition_numbers":      cond_values.cpu().numpy(),
        "worst_condition_number": worst_cond,
        "average_condition_number": avg_cond,
        # Hyperparameters
        "subset_batch_size": subset_batch_size,
        "condition_ridge":   condition_ridge,
        "eps":               eps,
        "lam_resid":         lam_resid,
        "lam_cond":          lam_cond,
        "lam_norm":          lam_norm,
    })
    return output_path, worst_cond, avg_cond


# %% Entry point — single seed run

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else None
    rc     = get_runtime_config(device)

    print("──────── Problem ────────")
    print(f"N (workers)       : {N}")
    print(f"(m, n, p)         : ({m}, {n}, {p})")
    print(f"thresh            : {thresh}   (= p·m·n + p − 1)")
    print(f"stragglers tol.   : {s_stragglers} (= N − thresh)")
    print(f"# targets         : {num_targets}  (one per α_{{j,k}})")
    print(f"# subsets         : {num_subsets}  (= C(N, thresh))")

    if device == "cuda":
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        print(f"CUDA runtime config: dtype={rc['dtype']}, "
              f"subset_batch_size={rc['subset_batch_size']}, "
              f"compile={enable_cuda_compile}, compile_mode={cuda_compile_mode}")
    else:
        print("Using CPU")

    # Pick the seeds for this sweep.
    if META_SEED is not None:
        random.seed(META_SEED)
    if NUM_SEEDS > SEED_POOL:
        raise ValueError(
            f"NUM_SEEDS={NUM_SEEDS} cannot exceed SEED_POOL={SEED_POOL} "
            f"(seeds are sampled without replacement)."
        )
    seeds = random.sample(range(1, SEED_POOL + 1), NUM_SEEDS)
    print(f"\nSweeping {NUM_SEEDS} seed(s): {seeds}")

    best_result = optimize_best_of_seeds(
        seeds=seeds,
        num_steps=NUM_STEPS,
        lr=LEARN_RATE,
        device=device,
        verbose=True,
        subset_batch_size=rc["subset_batch_size"],
    )

    X            = best_result["X"]
    Y            = best_result["Y"]
    history      = best_result["history"]
    step_times   = best_result["step_times"]
    best_seed    = best_result["seed"]
    seed_summary = best_result["summary"]

    final_objective = history[-1]
    best_objective  = min(history)

    mat_path, worst_cond, avg_cond = save_to_mat(
        X, Y, history,
        step_times=step_times,
        best_seed=best_seed,
        seed_summary=seed_summary,
        subset_batch_size=rc["subset_batch_size"],
    )

    # Per-seed leaderboard, sorted by final objective ascending
    print("\n──────── Per-seed summary (sorted) ────────")
    print(f"{'seed':>6} {'final_objective':>20} {'best_seen':>20}")
    for s, f, b in sorted(seed_summary, key=lambda r: r[1]):
        marker = "  <-- chosen" if s == best_seed else ""
        print(f"{s:>6d} {f:>20.6e} {b:>20.6e}{marker}")

    print("\n──────── Final result (best seed) ────────")
    print(f"Best seed:                   {best_seed}")
    print(f"Final-step objective:        {final_objective:.6e}")
    print(f"Best objective seen:         {best_objective:.6e}")
    print(f"Average step time:           {sum(step_times)/len(step_times):.3f}s")
    print(f"Worst-case condition number: {worst_cond:.6e}")
    print(f"Average condition number:    {avg_cond:.6e}")
    print(f"Saved MATLAB file:           {mat_path}")