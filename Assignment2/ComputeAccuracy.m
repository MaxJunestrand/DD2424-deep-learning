function acc = ComputeAccuracy(X, y, W, b)
    P = EvaluateClassifier(X, W, b);
    [~, prediction] = max(P);
    acc = length(find(y == prediction))/length(y);
end