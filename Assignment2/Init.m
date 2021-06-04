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

