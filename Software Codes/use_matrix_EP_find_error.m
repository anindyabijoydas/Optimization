clc
close all
clear

N = 11; m = 2; n = 2; p = 2; k = p*m*n + p - 1;
load coded_design_m2_n2_p2_N11_thresh9_torch_batched_result.mat

a_optimized = X;
b_optimized = Y;

big = 200;
NN = big*m*n*p;
AA = randn(NN,NN); 
BB = randn(NN,NN); 
CC = AA'*BB;

cc = big*m*n;
aa = big*n*p;
bb = big*m*p;

for i = 1:p
    for j = 1:m
        A{i,j} = AA((i-1)*cc+1:i*cc,(j-1)*aa+1:j*aa);
    end
end
for i = 1:p
    for j = 1:n
        B{i,j} = BB((i-1)*cc+1:i*cc,(j-1)*bb+1:j*bb);
    end
end

for i = 1:N
    suma = zeros(size(A{1,1}));
    sumb = zeros(size(B{1,1}));
    for j = 1:p
        for kk = 1:m
            suma = suma + a_optimized(i,j,kk)*A{j,kk};   
        end
    end
    for j = 1:p
        for kk = 1:n
            sumb = sumb + b_optimized(i,j,kk)*B{j,kk};   
        end
    end
    Wa{i} = suma;
    Wb{i} = sumb;
    Wab{i} = Wa{i}'*Wb{i};
end

sets = subset_matrix + 1;      
mm = size(sets,1);
err = zeros(mm,1);

for ss = 1:mm
    rec = zeros(size(CC));   
    for jj = 1:m
        for kk = 1:n
            rec_block = zeros(aa, bb);
            for ell = 1:k
                rec_block = rec_block + decoding_coeff(ss, ell, jj, kk)*Wab{sets(ss, ell)};
            end
            row = (jj-1)*aa + 1 : jj*aa;
            col = (kk-1)*bb + 1 : kk*bb;
            rec(row,col) = rec_block;
        end
    end
    err(ss) = norm(CC-rec)/norm(CC);
end

max_error = max(err)
mean_error = mean(err)