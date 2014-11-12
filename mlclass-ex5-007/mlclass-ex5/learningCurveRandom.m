function [error_train, error_val] = ...
    learningCurveRandom(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);


max_iter = 50;

for i= 1:m
	%theta_rand = zeros(max_iter, 1);
	
	error_train_rand = zeros(max_iter, 1);
	error_val_rand = zeros(max_iter, 1);
	for it=1:max_iter
		perm = randperm(m);
		X_rand = X(perm, :)(1:i, :);
		y_rand = y(perm)(1:i);
		theta = trainLinearReg(X_rand, y_rand, lambda);
		error_train_rand(it) = linearRegCostFunction(X_rand, y_rand, theta, 0);
		error_val_rand(it) = linearRegCostFunction(Xval, yval, theta, 0);
	endfor
	
	%theta = trainLinearReg(X(1:i, :), y(1:i), lambda);
	error_train(i) = mean(error_train_rand);
	error_val(i) = mean(error_val_rand);
endfor







% -------------------------------------------------------------

% =========================================================================

end
