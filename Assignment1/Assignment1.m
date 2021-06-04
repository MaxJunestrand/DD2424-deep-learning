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