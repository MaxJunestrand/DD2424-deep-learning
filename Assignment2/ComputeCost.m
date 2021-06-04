function [cost, loss] = ComputeCost(X, Y, W, b, lambda)
    P = EvaluateClassifier(X, W, b);
    loss = sum(-log(sum(Y.*P,1)))/size(X, 2);
    cost = loss + lambda*sum(sum(W{1}.^2))+ lambda*sum(sum(W{2}.^2));
end