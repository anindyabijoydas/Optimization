clc
close all
clear


N = 11; m = 2; n = 2; p = 2; k = p*m*n + p - 1;
load coded_design_m2_n2_p2_N11_thresh9_torch_batched_result.mat


a_optimized = X;            
b_optimized = Y;           
decoder_coeffs = decoding_coeff;
sets = subset_matrix + 1;

pm = p*m;
pn = p*n;

all_encoding_rows = zeros(N, pm*pn);

for i = 1:N
    xa = reshape(a_optimized(i,:,:), [pm, 1]);   
    yb = reshape(b_optimized(i,:,:), [pn, 1]);   
    all_encoding_rows(i,:) = kron(xa, yb);
end

mm = size(sets,1);
condition_numbers = zeros(mm,1);

for i = 1:mm
    M = all_encoding_rows(sets(i,:), :);   
    G = M * M';
    condition_numbers(i) = sqrt(cond(G));
end

max_cond = max(condition_numbers)
mean_cond = mean(condition_numbers)