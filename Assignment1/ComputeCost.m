function J = ComputeCost(X, Y, W, b, lambda)
%Uses one one Y data%
    P = EvaluateClassifier(X, W, b);
    lCross = sum(diag(-log(Y'*P)))/width(X);
    W2 = lambda*sum(sum(W.^2));
    J = lCross + W2;
end

