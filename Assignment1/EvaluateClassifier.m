function P = EvaluateClassifier(X, W, b)
% P has K x n size%
    s = W*X+b;
    P = softmax(s);
end

