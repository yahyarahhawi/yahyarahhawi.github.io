import torch
import random
class LinearModel:
    """
    A simple linear model for binary classification.
    """

    def __init__(self):
        self.w = None 


    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None:
            self.w = torch.rand((X.size(1)))
        return X @ self.w

    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        score = self.score(X)
        return (score > 0).float()
    
    

class MyLinearRegression(LinearModel):
    """
    A simple linear regression model for binary classification.
     """
    def __init__(self):
        super().__init__()

    def predict(self, X):
        return super().score(X)
        
    def loss(self, X, y):
        """
        Compute the mean squared error loss for the linear regression model.
        The loss is defined as:
        L(w) = 1/n * sum((y_i - X_i @ w)^2)
        where n is the number of samples, X_i is the i-th sample, and y_i is the corresponding label.
        The loss is averaged over all samples.
        """
        scores = self.score(X)
        return ((scores - y) ** 2).mean()
    

class OverParameterizedLinearRegressionOptimizer:


    def __init__(self, model):
        """
        Args:
            model: instance of LinearModel (or subclass)
        """
        self.model = model

    def fit(self, X, y):
        """
        Compute and set the optimal weight vector w = X⁺ y,
        where X⁺ = torch.linalg.pinv(X) is the Moore–Penrose pseudoinverse.

        After calling this, model.w contains the minimum‑norm least squares solution.
        """
        X_pinv = torch.linalg.pinv(X)
        w_opt = X_pinv @ y
        self.model.w = w_opt

        return self.model
