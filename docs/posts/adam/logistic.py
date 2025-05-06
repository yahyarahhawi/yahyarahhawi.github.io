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
        """
        if self.w is None:
            self.w = torch.randn(X.size(1)) * 0.01
        return X @ self.w

    def predict(self, X):
        """
        Compute binary predictions based on scores.
        """
        score = self.score(X)
        return (score > 0).float()


class LogisticRegression(LinearModel):
    """
    Logistic regression model for binary classification.
    Inherits from LinearModel.
    """

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def loss(self, X, y, w=None):
        """
        Binary cross-entropy loss.
        """
        if w is None:
            w = self.w if self.w is not None else torch.zeros(X.size(1))
        preds = torch.clamp(self.sigmoid(X @ w), 1e-7, 1 - 1e-7)
        return (-y * torch.log(preds) - (1 - y) * torch.log(1 - preds)).mean()

    def grad(self, X, y):
        """
        Compute the gradient of the logistic loss.
        """
        sigmoid = self.sigmoid(self.score(X))
        grad = (sigmoid - y).unsqueeze(1) * X  # shape: (n, p)
        return grad.mean(dim=0)  # shape: (p,)


class GradientDescentOptimizer:
    """
    Gradient descent with optional momentum.
    """

    def __init__(self, model):
        self.model = model
        self.prev_w = None

    def step(self, X, y, alpha=0.1, beta=0.0):
        if self.model.w is None:
            self.model.w = torch.randn(X.size(1)) * 0.01

        grad = self.model.grad(X, y)

        if self.prev_w is None:
            self.prev_w = self.model.w.clone()

        momentum = beta * (self.model.w - self.prev_w)
        new_w = self.model.w - alpha * grad + momentum

        self.prev_w = self.model.w.clone()
        self.model.w = new_w


class NewtonOptimizer():
    """
    Newton's method optimizer for the logistic regression model.
    """
    def __init__(self, model):
        self.model = model
        self.prev_w = None  

    def step(self, X, y, alpha=1.0):
        """
        Compute one step of Newton's method:
        w_new = w_old - alpha * H^{-1} * grad
        """
        if self.model.w is None:
            self.model.w = torch.randn(X.size(1)) * 0.01

        grad = self.model.grad(X, y)
        hessian = self.hessian(X, y)

        hessian_inv = torch.linalg.inv(hessian)
        new_w = self.model.w - alpha * (hessian_inv @ grad)

        self.prev_w = self.model.w.clone()
        self.model.w = new_w
        
    def hessian(self, X, y):
        """
        Compute the Hessian matrix H(w) = (1/n) * X^T D X
        where D is a diagonal matrix with entries: σ(s_k)(1 - σ(s_k))
        A small ε is added to the diagonal for numerical stability.
        """
        scores = self.model.score(X)
        probs = self.model.sigmoid(scores)
        diag = probs * (1 - probs)
        D = torch.diag(diag)

        H = X.T @ D @ X
        H = H / X.size(0)  # Normalize by number of samples

        return H
    

class AdamOptimizer():
    """
    Adam optimizer for the logistic regression model.
    """
    def __init__(self, model, alpha=0.001, beta_1=0.9, beta_2=0.999, batch_size=None, w_0=None):
        self.model = model
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.batch_size = batch_size
        self.t = 0  # timestep
        self.m = None #these are the first moment estimates
        self.v = None #this is the second moment estimates

        if w_0 is not None:
            self.model.w = w_0.clone()

    def step(self, X, y):
        """
        Perform one step of the Adam update.
        If batch_size is set, a random mini-batch is sampled from (X, y).
        """
        n = X.size(0)
        self.t += 1

        # Mini-batch selection
        if self.batch_size is not None and self.batch_size < n:
            indices = torch.randperm(n)[:self.batch_size]
            X_batch = X[indices]
            y_batch = y[indices]
        else:
            X_batch = X
            y_batch = y

        grad = self.model.grad(X_batch, y_batch)

        if self.m is None:
            self.m = torch.zeros_like(grad)
            self.v = torch.zeros_like(grad)

        # Update biased first and second moment estimates
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (grad ** 2)

        # Bias correction
        m_hat = self.m / (1 - self.beta_1 ** self.t)
        v_hat = self.v / (1 - self.beta_2 ** self.t)

        # Update weights
        self.model.w = self.model.w - self.alpha * m_hat / (torch.sqrt(v_hat) + 1e-8)
    

