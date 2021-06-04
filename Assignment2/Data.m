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

