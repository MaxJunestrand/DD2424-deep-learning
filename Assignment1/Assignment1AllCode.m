function [X, Y, y] = LoadBatch(filename)
% x pixel data %
% Y on hot labels for each image%
% y labels for all data %
    A = load(filename);
    X = double(A.data');
    y = double(A.labels')+1;
    Y = onehotencode(categorical(A.labels),2)';
end


function [trainX, valX, testX] = Normalize(trainX, valX, testX)
    mean_X = mean(trainX, 2);
    std_X = std(trainX, 0, 2);
    
    % train
    trainX = trainX - repmat(mean_X, [1, size(trainX, 2)]);
    trainX = trainX ./ repmat(std_X, [1, size(trainX, 2)]);
    % validation
    valX = valX - repmat(mean_X, [1, size(valX , 2)]);
    valX = valX ./ repmat(std_X, [1, size(valX , 2)]);
    % test
    testX = testX - repmat(mean_X, [1, size(testX , 2)]); 
    testX = testX ./ repmat(std_X, [1, size(testX , 2)]);
end


function [W,b] = Init(K,d)
%INIT Summary of this function goes here
%   Detailed explanation goes here
s = 0.01;
W = randn(K, d)*s;
b = randn(K, 1)*s;
end

function GDObject = CreateGDObject(n_batch, eta, n_epochs)
    GDObject.n_batch = n_batch;
    GDObject.eta = eta;
    GDObject.n_epochs = n_epochs;
end

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


function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

no = size(W, 1);
d = size(X, 1);

grad_W = zeros(size(W));
grad_b = zeros(no, 1);

for i=1:length(b)
    b_try = b;
    b_try(i) = b_try(i) - h;
    c1 = ComputeCost(X, Y, W, b_try, lambda);
    b_try = b;
    b_try(i) = b_try(i) + h;
    c2 = ComputeCost(X, Y, W, b_try, lambda);
    grad_b(i) = (c2-c1) / (2*h);
end

for i=1:numel(W)
    
    W_try = W;
    W_try(i) = W_try(i) - h;
    c1 = ComputeCost(X, Y, W_try, b, lambda);
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    c2 = ComputeCost(X, Y, W_try, b, lambda);
    
    grad_W(i) = (c2-c1) / (2*h);
end



function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)

no = size(W, 1);
d = size(X, 1);

grad_W = zeros(size(W));
grad_b = zeros(no, 1);

c = ComputeCost(X, Y, W, b, lambda);

for i=1:length(b)
    b_try = b;
    b_try(i) = b_try(i) + h;
    c2 = ComputeCost(X, Y, W, b_try, lambda);
    grad_b(i) = (c2-c) / h;
end

for i=1:numel(W)   
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    c2 = ComputeCost(X, Y, W_try, b, lambda);
    
    grad_W(i) = (c2-c) / h;
end

function [grad_W,grad_b] = ComputeGradients(X, Y, P, W, lambda)

grad_W = zeros(size(W));
grad_b = zeros(size(W, 1), 1);

% From last slides in lecture 3 %
for i = 1 : width(X)
    Pi = P(:, i);
    Yi = Y(:, i);
    Xi = X(:, i);
    g = -(Yi-Pi);
    g2 = -Yi'*(diag(Pi) - Pi*Pi')/(Yi'*Pi);
    grad_b = grad_b + g;
    grad_W = grad_W + g*Xi';
end

grad_b = grad_b/width(X);
grad_W = 2*lambda*W + grad_W/width(X);

end

function P = EvaluateClassifier(X, W, b)
% P has K x n size%
    s = W*X+b;
    P = softmax(s);
end


function J = ComputeCost(X, Y, W, b, lambda)
%Uses one one Y data%
    P = EvaluateClassifier(X, W, b);
    lCross = sum(diag(-log(Y'*P)))/width(X);
    W2 = lambda*sum(sum(W.^2));
    J = lCross + W2;
end

function acc = ComputeAccuracy(X, y, W, b)
    P = EvaluateClassifier(X, W, b);
    [~, prediction] = max(P);
    acc = length(find(y == prediction))/length(y);
end

%THIS WAS THE RUNNER CODE. EVERYTHING ABOVE THIS IS FUNCTIONS TAKEN FROM
%OTHER FILES.
clear;
addpath Datasets/cifar-10-batches-mat/;

% Step 1%
[trainX, trainY, trainy ] = LoadBatch('data_batch_1.mat');
[valX, valY, valy ] = LoadBatch('data_batch_2.mat');
[testX, testY, testy ] = LoadBatch('test_batch.mat');

% Step 2%
[trainX, valX, testX] = Normalize(trainX, valX, testX);

% Step 3%
K = size(trainY, 1);
d = size(trainX, 1);
[W, b] = Init(K, d);

% check to see that the analytical gradients are correct by checking
% relative result with existing functions.
batch_size = 1;
lambda = 0.1;
[ngrad_b, ngrad_W] = ComputeGradsNumSlow(trainX(:, 1 : batch_size), trainY(:, 1 : batch_size), W, b, lambda, 1e-6);
P = EvaluateClassifier(trainX(:, 1 : batch_size), W, b);
[grad_W, grad_b] = ComputeGradients(trainX(:, 1 : batch_size), trainY(:, 1 : batch_size), P, W, lambda);
gradcheck_b = max(abs(ngrad_b - grad_b)./max(0.00001, abs(ngrad_b) + abs(grad_b)));
gradcheck_W = max(max(abs(ngrad_W - grad_W)./max(0.00001, abs(ngrad_W) + abs(grad_W))));

% Step 4%
GDObject = CreateGDObject(100, 0.001, 40);
n_epochs = GDObject.n_epochs;
%lambda = 0.1;
%lambda = 1;
%GDObject = GDObject(100, 0.001, 40);
lossTraining = zeros(1, n_epochs );
lossValidation = zeros(1, n_epochs );
for i = 1 : n_epochs 
    lossTraining(i) = ComputeCost(trainX, trainY, W, b, lambda);
    lossValidation(i) = ComputeCost(valX, valY, W, b, lambda);
    [W, b] = MiniBatchGD(trainX, trainY, GDObject, W, b, lambda);
end

% step 5
acc_training = ComputeAccuracy(trainX, trainy, W, b);
disp(['training accuracy:' num2str(acc_training*100) '%'])
acc_test = ComputeAccuracy(testX, testy, W, b);
disp(['test accuracy:' num2str(acc_test*100) '%'])

% plot cost score
figure()
plot(1 : n_epochs , lossTraining, 'r')
hold on
plot(1 : n_epochs , lossValidation, 'b')
hold off
xlabel('epoch');
ylabel('loss');
legend('training loss', 'validation loss');

% code from assignment to print pictures of W
for i = 1 : K
    im = reshape(W(i, :), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
end

figure()
montage(s_im, 'size', [1, K])