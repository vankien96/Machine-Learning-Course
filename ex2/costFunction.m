function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
sum = 0;
gradSum = 0;
for i=1:m
    mul = X(i, :) * theta;
    hx = sigmoid(mul);
    
    sum = sum + (- y(i,1)*log(hx) - (1 - y(i,1))*log(1 - hx));
    
    gradSum = gradSum + (hx - y(i,1))*X(i, :);
end

J = sum / m;
grad = gradSum / m;

% =============================================================

end
