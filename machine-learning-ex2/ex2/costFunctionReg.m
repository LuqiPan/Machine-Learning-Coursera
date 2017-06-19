function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



theta_squared = 0;
for j = 1:size(theta, 1)
  if (j != 1)
    theta_squared = theta_squared + theta(j) * theta(j);
  end
end

for i = 1:m
  h_theta = sigmoid(X(i,:) * theta);
  J = J + y(i) * log(h_theta) + (1 - y(i)) * log(1 - h_theta);
end

J = -J / m + lambda * theta_squared / 2 / m;

for j = 1:size(theta)
  temp = 0;
  for i = 1:m
    h_theta = sigmoid(X(i,:) * theta);
    temp = temp + (h_theta - y(i)) * X(i,j);
  end
  if (j == 1)
    grad(j) = temp / m;
  else
    grad(j) = temp / m + lambda * theta(j) / m;
  end
end



% =============================================================

end
