%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Driver Code%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clear;
% addpath Datasets/cifar-10-batches-mat/;
% 
% % Start of Question 1%
% [trainX, trainY, trainy ] = LoadBatch('data_batch_1.mat');
% [valX, valY, valy ] = LoadBatch('data_batch_2.mat');
% [testX, testY, testy ] = LoadBatch('test_batch.mat');
% 
% [trainX, valX, testX] = Normalize(trainX, valX, testX);
% 
% d = size(trainX, 1);
% K = size(trainY, 1);
% [W, b] = Init(K, d);
% 
% batch_size = 20;
% lambda = 0;
% Xcheck = trainX(:, 1 : batch_size);
% Ycheck = trainY(:, 1 : batch_size);
% %numerical gradients
% [ngrad_b, ngrad_W] = ComputeGradsNum(Xcheck, Ycheck, W, b, lambda, 1e-5);
% 
% %analytical gradients
% P = EvaluateClassifier(Xcheck, W, b);
% [grad_W, grad_b] = ComputeGradients(Xcheck, Ycheck, P, W, b, lambda);
% 
% %relative error rate to check that the gradients are calculated correctly.
% eps = 0.001;
% 
% gradcheck_b1 = sum(abs(ngrad_b{1} - grad_b{1})/max(eps, sum(abs(ngrad_b{1}) + abs(grad_b{1}))));
% gradcheck_W1 = sum(sum(abs(ngrad_W{1} - grad_W{1})/max(eps, sum(sum(abs(ngrad_W{1}) + abs(grad_W{1}))))));
% gradcheck_b2 = sum(abs(ngrad_b{2} - grad_b{2})/max(eps, sum(abs(ngrad_b{2}) + abs(grad_b{2}))));
% gradcheck_W2 = sum(sum(abs(ngrad_W{2} - grad_W{2})/max(eps, sum(sum(abs(ngrad_W{2}) + abs(grad_W{2}))))));
% gradcheck_W = [gradcheck_W1, gradcheck_W2];
% gradcheck_b = [gradcheck_b1, gradcheck_b2];

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




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%FUNCTIONS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [lossTraining, lossValidation, costTraining, costValidation, accTraining, accValidation] = MiniBatchGD(X, Y, y, GDparams, W, b, lambda, eta_min, eta_max, n_s, valX, valY, valy)
    % from previous
    n = size(X, 2);
    n_batch = GDparams.n_batch;
    n_epochs = GDparams.n_epochs;
    eta = eta_min;
    t = 0;
    lossTraining = zeros(1, n_epochs );
    lossValidation = zeros(1, n_epochs );
    costTraining = zeros(1, n_epochs ); 
    costValidation = zeros(1, n_epochs );
    accTraining = zeros(1, n_epochs ); 
    accValidation = zeros(1, n_epochs );
    for i = 1 : n_epochs
        
        [trainCost, trainLoss] = ComputeCost(X, Y, W, b, lambda);
        [valCost, valLoss] = ComputeCost(valX, valY, W, b, lambda);

        lossTraining(1, i) = trainLoss;
        lossValidation(1, i) = valLoss;

        costTraining(1, i) = trainCost;
        costValidation(1, i) = valCost;

        accTraining(1, i) = ComputeAccuracy(X, y, W, b);
        accValidation(1, i) = ComputeAccuracy(valX, valy, W, b);
        
        for j = 1:(n_batch-1)
            % Code from assignment that gets image batches
            j_start = floor(j*(n/n_batch));
            j_end = floor((j+1)*(n/n_batch));
            inds = j_start:j_end;
            Xbatch = X(:, inds);
            Ybatch = Y(:, inds);

            % gradient
            P = EvaluateClassifier(Xbatch, W, b);
            [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, b, lambda);

            % weights & bias
                W{1} = W{1} - eta * grad_W{1};
                W{2} = W{2} - eta * grad_W{2};
                b{1} = b{1} - eta * grad_b{1};
                b{2} = b{2} - eta * grad_b{2};
            
            % Cyclic learning rate   
            if (t <= n_s)                    
                %eta = eta_min + ((t - 2*n_s*lambda)/n_s)*(eta_max - eta_min);
                eta = eta_min + t/n_s * (eta_max - eta_min);
            elseif (t <= 2*n_s)
                %eta = eta_max - ((t - (2*lambda+1)*n_s)/n_s)*(eta_max - eta_min);
                eta = eta_max - (t - n_s)/n_s * (eta_max - eta_min);
            end
            t = mod((t+1), (2*n_s));
           
        end         
            
    end
end


function acc = ComputeAccuracy(X, y, W, b)
    P = EvaluateClassifier(X, W, b);
    [~, prediction] = max(P);
    acc = length(find(y == prediction))/length(y);
end

function [cost, loss] = ComputeCost(X, Y, W, b, lambda)
    P = EvaluateClassifier(X, W, b);
    loss = sum(-log(sum(Y.*P,1)))/size(X, 2);
    cost = loss + lambda*sum(sum(W{1}.^2))+ lambda*sum(sum(W{2}.^2));
end

function [grad_W,grad_b] = ComputeGradients(X, Y, P, W, b, lambda)
n = size(X,2);
% Slide 46 shows all the steps
g = -(Y-P)';
grad_b2=(sum(g,1)/n)';
s1 = W{1}*X+repmat(b{1},1,n);
h = max(0, s1);
grad_W2 = g'*h'/n+2*lambda*W{2};

% Relu
g=g*W{2};
g=g.*(s1>0)'; 

grad_b1=(sum(g,1)/n)';
grad_W1 = g'*X'/n+2*lambda*W{1};

% Results
grad_W={grad_W1,grad_W2};
grad_b={grad_b1,grad_b2};
end

function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

cost = ComputeCost(X, Y, W, b, lambda);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        grad_b{j}(i) = (c2-cost) / h;
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})   
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);
        
        grad_W{j}(i) = (c2-cost) / h;
    end
end

function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) - h;
        cost1 = ComputeCost(X, Y, W, b_try, lambda);
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        cost2 = ComputeCost(X, Y, W, b_try, lambda);
        
        grad_b{j}(i) = (cost2-cost1) / (2*h);
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})
        
        W_try = W;
        W_try{j}(i) = W_try{j}(i) - h;
        cost1 = ComputeCost(X, Y, W_try, b, lambda);
    
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        cost2 = ComputeCost(X, Y, W_try, b, lambda);
    
        grad_W{j}(i) = (cost2-cost1) / (2*h);
    end
end


function GDObject = CreateGDObject(n_batch, n_epochs)
    GDObject.n_batch = n_batch;
    GDObject.n_epochs = n_epochs;
end

function [data, labels] = Data(nr_of_validation)
%DATA Summary of this function goes here
%   Detailed explanation goes here
[X1, Y1, y1 ] = LoadBatch('data_batch_1.mat');
[X2, Y2, y2 ] = LoadBatch('data_batch_2.mat');
[X3, Y3, y3 ] = LoadBatch('data_batch_3.mat');
[X4, Y4, y4 ] = LoadBatch('data_batch_4.mat');
[X5, Y5, y5 ] = LoadBatch('data_batch_5.mat');

% Concat matrices
trainX = [X1, X2, X3, X4, X5];
trainY = [Y1, Y2, Y3, Y4, Y5];
trainy = [y1, y2, y3, y4, y5];

valX = trainX(:, 1:nr_of_validation);
valY = trainY(:, 1:nr_of_validation);
valy = trainy(:, 1:nr_of_validation);

trainX = trainX(:, nr_of_validation:end);
trainY = trainY(:, nr_of_validation:end);
trainy = trainy(:, nr_of_validation:end);

[testX, testY, testy ] = LoadBatch('test_batch.mat');

data.trainX = trainX;
data.trainY = trainY;
data.trainy = trainy;

data.valX = valX;
data.valY = valY;
data.valy = valy;

data.testX = testX;
data.testY = testY;
data.testy = testy;

[data.trainX, data.valX, data.testX] = Normalize(data.trainX, data.valX, data.testX);

labels = load('batches.meta.mat');
end



function [P, h] = EvaluateClassifier(X, W, b)
% P has K x n size%
    n = size(X,2);
    W1 = W{1};
    W2 = W{2};
    b1 = b{1};
    b1 = repmat(b1,1,n);
    b2 = b{2};
    b2 = repmat(b2,1,n);
    s1 = W1*X+b1;
    h = max(0, s1);
    s = W2*h+b2;
    P = softmax(s);
end



function [W,b] = Init(K,d)
%INIT Summary of this function goes here
%   Detailed explanation goes here
m = 50;
W1 = 1/sqrt(d)*randn(m, d);
W2 = 1/sqrt(m)*randn(K, m);
b1 = zeros(m, 1);
b2 = zeros(K, 1);
W = {W1, W2};
b = {b1, b2};
end


function [X, Y, y] = LoadBatch(filename)
% x pixel data %
% Y on hot labels for each image%
% y labels for all data %
    A = load(filename);
    X = double(A.data');
    y = double(A.labels')+1;
    Y = onehotencode(categorical(A.labels),2)';
end



function [lossTraining, lossValidation, costTraining, costValidation, W, b] = MiniBatchGD(X, Y, y, GDparams, W, b, lambda, eta_min, eta_max, n_s, valX, valY, valy)
    % from previous
    n = size(X, 2);
    n_batch = GDparams.n_batch;
    n_epochs = GDparams.n_epochs;
    eta = eta_min;
    t = 0;
    lossTraining = zeros(1, n_epochs );
    lossValidation = zeros(1, n_epochs );
    costTraining = zeros(1, n_epochs ); 
    costValidation = zeros(1, n_epochs );
    for i = 1 : n_epochs
        for j = 1:(n_batch-1)
            % Code from assignment that gets image batches
            j_start = floor(j*(n/n_batch));
            j_end = floor((j+1)*(n/n_batch));
            inds = j_start:j_end;
            Xbatch = X(:, inds);
            Ybatch = Y(:, inds);

            % gradient
            P = EvaluateClassifier(Xbatch, W, b);
            [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, b, lambda);

            % weights & bias
                W{1} = W{1} - eta * grad_W{1};
                W{2} = W{2} - eta * grad_W{2};
                b{1} = b{1} - eta * grad_b{1};
                b{2} = b{2} - eta * grad_b{2};
            
            % Cyclic learning rate   
            if (t <= n_s)                    
                %eta = eta_min + ((t - 2*n_s*lambda)/n_s)*(eta_max - eta_min);
                eta = eta_min + t/n_s * (eta_max - eta_min);
            elseif (t <= 2*n_s)
                %eta = eta_max - ((t - (2*lambda+1)*n_s)/n_s)*(eta_max - eta_min);
                eta = eta_max - (t - n_s)/n_s * (eta_max - eta_min);
            end
            t = mod((t+1), (2*n_s));
           
        end
        
            Wstar = W;
            bstar = b;
            
            [trainCost, trainLoss] = ComputeCost(X, Y, W, b, lambda);
            [valCost, valLoss] = ComputeCost(valX, valY, W, b, lambda);
            
            lossTraining(1, i) = trainLoss;
            lossValidation(1, i) = valLoss;
            
            costTraining(1, i) = trainCost;
            costValidation(1, i) = valCost;
    end
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

function [] = PlotFigure(nr, n_epochs, train, val, x_label, y_label, y_max)
figure(nr)
plot(1 : n_epochs , train, 'r')
hold on
plot(1 : n_epochs , val, 'b')
ylim([0 y_max]);
hold off
xlabel(x_label);
ylabel(y_label);
legend('training', 'validation');
end

















