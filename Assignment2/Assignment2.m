clear;
addpath Datasets/cifar-10-batches-mat/;

% Start of Question 1%
[trainX, trainY, trainy ] = LoadBatch('data_batch_1.mat');
[valX, valY, valy ] = LoadBatch('data_batch_2.mat');
[testX, testY, testy ] = LoadBatch('test_batch.mat');

[trainX, valX, testX] = Normalize(trainX, valX, testX);

d = size(trainX, 1);
K = size(trainY, 1);
[W, b] = Init(K, d);

batch_size = 20;
lambda = 0;
Xcheck = trainX(:, 1 : batch_size);
Ycheck = trainY(:, 1 : batch_size);
%numerical gradients
[ngrad_b, ngrad_W] = ComputeGradsNum(Xcheck, Ycheck, W, b, lambda, 1e-5);

%analytical gradients
P = EvaluateClassifier(Xcheck, W, b);
[grad_W, grad_b] = ComputeGradients(Xcheck, Ycheck, P, W, b, lambda);

%relative error rate to check that the gradients are calculated correctly.
eps = 0.001;

gradcheck_b1 = sum(abs(ngrad_b{1} - grad_b{1})/max(eps, sum(abs(ngrad_b{1}) + abs(grad_b{1}))));
gradcheck_W1 = sum(sum(abs(ngrad_W{1} - grad_W{1})/max(eps, sum(sum(abs(ngrad_W{1}) + abs(grad_W{1}))))));
gradcheck_b2 = sum(abs(ngrad_b{2} - grad_b{2})/max(eps, sum(abs(ngrad_b{2}) + abs(grad_b{2}))));
gradcheck_W2 = sum(sum(abs(ngrad_W{2} - grad_W{2})/max(eps, sum(sum(abs(ngrad_W{2}) + abs(grad_W{2}))))));
gradcheck_W = [gradcheck_W1, gradcheck_W2];
gradcheck_b = [gradcheck_b1, gradcheck_b2];

% END of Question 1%

% Start of Question 2%

% GDObject = CreateGDObject(100, 48); % batch size of descent | nr of epochs
% n_epochs = GDObject.n_epochs;
% eta_min = 1e-5;
% eta_max = 1e-1;
% n_s = 800;
% lambda = 0.01;
% 
% [lossTraining, lossValidation, costTraining, costValidation, accTraining, accValidation] = AccMiniBatchGD(trainX, trainY, trainy, GDObject, W, b, lambda, eta_min, eta_max, n_s, valX, valY, valy);
% PlotFigure(1, n_epochs, costTraining, costValidation,"nr epochs", "Cost Plot", 4); % Cost
% PlotFigure(2, n_epochs, lossTraining, lossValidation,"nr epochs", "Loss Plot", 2.5); % Loss
% PlotFigure(3, n_epochs, accTraining, accValidation,"nr epochs", "Acc Plot", 1); % Cost

% End of Question 2

% % Start of Question 3
% % Constants:
% l_min = -5;
% l_max = -1;
% eta_min = 1e-5;
% eta_max = 1e-1;
% [data, labels] = Data(5000);
% GDObject = CreateGDObject(100, 36); % batch size of descent | nr of epochs
% 
% n = size(data.trainX,2);
% n_s = 2*floor(n / GDObject.n_batch);
% accuracies = zeros(1, 8);
% lambdas = zeros(1, 8);
% for i=1:8
%     d = size(data.trainX, 1);
%     K = size(data.trainY, 1);
%     [W, b] = Init(K, d);
% 
%     l = l_min + (l_max - l_min)*rand(1, 1);
%     lambda = 10^l;
%     
%     [lossTraining, lossValidation, costTraining, costValidation, W, b] = MiniBatchGD(data.trainX, data.trainY, data.trainy, GDObject, W, b, lambda, eta_min, eta_max, n_s, data.valX, data.valY, data.valy); 
%     lambdas(1, i) = lambda;
%     accuracies(1, i) = ComputeAccuracy(data.valX, data.valy, W, b);
% end
% 
% % End of Question 3
% 
% % Start of Question 4
% % Constants:
% l_min = -3;
% l_max = -2;
% eta_min = 1e-5;
% eta_max = 1e-1;
% [data, labels] = Data(5000);
% GDObject = CreateGDObject(100, 36); % batch size of descent | nr of epochs
% 
% n = size(data.trainX,2);
% n_s = 2*floor(n / GDObject.n_batch);
% accuracies = zeros(1, 8);
% lambdas = zeros(1, 8);
% for i=1:8
%     d = size(data.trainX, 1);
%     K = size(data.trainY, 1);
%     [W, b] = Init(K, d);
% 
%     l = l_min + (l_max - l_min)*rand(1, 1);
%     lambda = 10^l;
%     
%     [lossTraining, lossValidation, costTraining, costValidation, W, b] = MiniBatchGD(data.trainX, data.trainY, data.trainy, GDObject, W, b, lambda, eta_min, eta_max, n_s, data.valX, data.valY, data.valy); 
%     lambdas(1, i) = lambda;
%     accuracies(1, i) = ComputeAccuracy(data.valX, data.valy, W, b);
% end

% End of Question 4

% % Start of Question 5
% % Constants:
% l_min = -3;
% l_max = -2;
% eta_min = 1e-5;
% eta_max = 1e-1;
% [data, labels] = Data(1000);
% GDObject = CreateGDObject(100, 40); % batch size of descent | nr of epochs
% 
% n = size(data.trainX,2);
% n_s = 2*floor(n / GDObject.n_batch);
% d = size(data.trainX, 1);
% K = size(data.trainY, 1);
% [W, b] = Init(K, d);
% lambda = 0.001013229;
% 
% [lossTraining, lossValidation, costTraining, costValidation, W, b] = MiniBatchGD(data.trainX, data.trainY, data.trainy, GDObject, W, b, lambda, eta_min, eta_max, n_s, data.valX, data.valY, data.valy);
% accuracy = ComputeAccuracy(data.valX, data.valy, W, b);
% 
% %PlotFigure(1, n_epochs, costTraining, costValidation,"nr epochs", "Loss Plot", 4); % Cost
% %PlotFigure(2, n_epochs, lossTraining, lossValidation,"nr epochs", "Loss Plot", 2.5); % Loss
% %PlotFigure(3, n_epochs, accTraining, accValidation,"nr epochs", "Loss Plot", 1); % Cost
% % End of Question 5



