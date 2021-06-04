function [X, Y, y] = LoadBatch(filename)
% x pixel data %
% Y on hot labels for each image%
% y labels for all data %
    A = load(filename);
    X = double(A.data');
    y = double(A.labels')+1;
    Y = onehotencode(categorical(A.labels),2)';
end
