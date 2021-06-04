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



