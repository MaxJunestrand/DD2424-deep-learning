function [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda)

n = size(X, 2);
n_batch = GDparams.n_batch;
eta = GDparams.eta;
for j = 1:(n/n_batch)
    % Code from assignment that gets image batches
    j_start = (j-1)*n_batch +1;
    j_end = j*n_batch;
    inds = j_start:j_end;
    Xbatch = X(:, inds);
    Ybatch = Y(:, inds);
   
    % gradient
    P = EvaluateClassifier(Xbatch, W, b);
    [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, lambda);
    
    % weights & bias
    W = W - eta*grad_W;
    b = b - eta*grad_b;
end
Wstar = W;
bstar = b;
end

