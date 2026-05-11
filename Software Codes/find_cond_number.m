clc
close all
clear

n = 7; m = 3; k = 5;
load coded_design_5_of_7_torch_batched_result_1e-06_1e-08.mat

a_optimized = X;
b_optimized = Y;
decoder_coeffs = decoding_coeff;
sets = subset_matrix+1;

all_encoding_rows = zeros(n,m*m);

for i = 1:n
    all_encoding_rows(i,:) = kron(a_optimized(i,:),b_optimized(i,:));
end

mm = size(sets,1);
condition_numbers = zeros(mm,1);
for i = 1:mm
    condition_numbers(i) = cond(all_encoding_rows(sets(i,:),:));
end
max_cond = max(condition_numbers)
mean_cond = mean(condition_numbers)
