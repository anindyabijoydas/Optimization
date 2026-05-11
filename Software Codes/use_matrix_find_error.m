clc
close all
clear

n = 7; m = 3; k = 5;
load coded_design_5_of_7_torch_batched_result_1e-06_1e-08.mat

a_optimized = X;
b_optimized = Y;
decoder_coeffs = decoding_coeff;
sets = subset_matrix+1;

big = 20;
N = big*m;
AA = randn(N,N); 
BB = randn(N,N); 
CC = AA*BB;

for i = 1:m
    A{i} = AA(:,(i-1)*big+1:i*big);
    B{i} = BB((i-1)*big+1:i*big,:);
end


for i = 1:n
    suma = zeros(size(A{1}));
    sumb = zeros(size(B{1}));
    for j = 1:m
        suma = suma + a_optimized(i,j)*A{j};   
        sumb = sumb + b_optimized(i,j)*B{j};
    end
    Wa{i} = suma;
    Wb{i} = sumb;
    Wab{i} = Wa{i}*Wb{i};
end

mm = size(sets,1);
err = zeros(mm,1);
r = size(A{1},1);
t = size(B{1},2);

for i = 1:mm
    rec = zeros(r,t);
    for j = 1:k
        rec = rec + decoder_coeffs(i,j)*Wab{sets(i,j)};
    end
    err(i) = norm(CC-rec)/norm(CC);
end
max_error = max(err)
mean_error = mean(err)