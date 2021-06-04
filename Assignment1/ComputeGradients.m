function [grad_W,grad_b] = ComputeGradients(X, Y, P, W, lambda)

grad_W = zeros(size(W));
grad_b = zeros(size(W, 1), 1);

% From last slides in lecture 3 %
for i = 1 : width(X)
    Pi = P(:, i);
    Yi = Y(:, i);
    Xi = X(:, i);
    g = -(Yi-Pi);
    g2 = -Yi'*(diag(Pi) - Pi*Pi')/(Yi'*Pi);
    grad_b = grad_b + g;
    grad_W = grad_W + g*Xi';
end

grad_b = grad_b/width(X);
grad_W = 2*lambda*W + grad_W/width(X);

end

