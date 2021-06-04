function [W,b] = Init(K,d)
%INIT Summary of this function goes here
%   Detailed explanation goes here
s = 0.01;
W = randn(K, d)*s;
b = randn(K, 1)*s;
end

