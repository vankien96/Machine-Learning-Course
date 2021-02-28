function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

sum = 0;
gradSum = 0;
for i=1:m
    mul = X(i, :) * theta;
    hx = sigmoid(mul);
    
    sum = sum + (- y(i,1)*log(hx) - (1 - y(i,1))*log(1 - hx));
    
    gradSum = gradSum + (hx - y(i,1))*X(i, :);
end

thetaSumSquare = 0;
for j=2:n
    thetaSumSquare = thetaSumSquare + theta(j) ^ 2;
end

thetaSumSquare = lambda / (2 * m) * thetaSumSquare;

J = sum / m + thetaSumSquare;
grad = gradSum / m;
for k=2:n
    grad(k) = grad(k) + lambda * theta(k) / m;
end

% =============================================================

end
