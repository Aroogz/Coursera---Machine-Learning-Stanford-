function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

testval =0.01*(3.^[0:8]);
params = zeros(1,2); %initializes parameter
error = 10; %sets error to an initial unlikely value so it gets replaced at loop initial
for i = 1:length(testval)
    for j = 1:length(testval)
        params_new = [testval(i),testval(j)];
        %get the model trained with the given set of parameters
        model= svmTrain(X, y, testval(i), @(x1, x2) gaussianKernel(x1, x2, testval(j)));
        %the predictions for given cross validation samples
        prediction = svmPredict(model, Xval);
        
        %get the fraction of predictions gotten correctly
        error_new = mean(double(prediction ~= yval));
        if error_new < error
            params = params_new;
            error = error_new;
        end
    end
        
end
C = params(1);
sigma = params(2);




% =========================================================================

end
