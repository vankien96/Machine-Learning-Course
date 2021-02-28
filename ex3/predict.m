function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

numActivation = size(Theta1, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
oneRow = ones(size(X, 1), 1);
X = [oneRow X];

for i = 1:m
    a = zeros(numActivation + 1, 1);
    a(1) = 1;
    for j = 1:numActivation
        theta = Theta1(j, :);
        a(j + 1) = sigmoid(theta * X(i, :)');
    end
    max = 0;
    for k = 1:num_labels
        theta = Theta2(k, :);
        predict = sigmoid(theta * a);
        
        if predict > max
            max = predict;
            p(i) = k;
        end
    end
end
% =========================================================================


end
