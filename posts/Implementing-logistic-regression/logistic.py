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
    
class LogisticRegression(LinearModel):
    """
    Logistic regression model for binary classification.
    Inherits from LinearModel.
    """
    def __init__(self):
        super().__init__()
        self.population = []

    def sigmoid(self, x):
        """
        Compute the sigmoid function for each element in x.
        """
        return 1 / (1 + torch.exp(-x))

    def loss(self, X, y, w=None):
        """
        Compute the binary cross-entropy loss for the logistic regression model.
        The loss is defined as:
        L(w) = -1/n * sum(y_i * log(sigmoid(X_i @ w)) + (1 - y_i) * log(1 - sigmoid(X_i @ w)))
        where n is the number of samples, X_i is the i-th sample, and y_i is the corresponding label.
        The loss is averaged over all samples.
        If w is not provided, the model's current weights are used.
        """
        if w is None:
            w = self.w
        preds = torch.clamp(self.sigmoid(X @ w), 1e-7, 1 - 1e-7) # Avoid log(0)
        return (-y * torch.log(preds) - (1 - y) * torch.log(1 - preds)).mean()
    
    def grad(self, X, y):
        """
        Compute the gradient of the loss function with respect to the weights w.
        The gradient is defined as:
        grad(w) = 1/n * sum((sigmoid(X_i @ w) - y_i) * X_i)
        where n is the number of samples, X_i is the i-th sample, and y_i is the corresponding label.
        The gradient is averaged over all samples.
        """
        sigmoid = self.sigmoid(self.score(X))
        grad = (sigmoid - y)[:, None] * X
        return grad.mean(0)
    

class GradientDescentOptimizer():
    """
    Gradient descent optimizer for the logistic regression model.
    """
    def __init__(self, model):
        self.model = model
        self.prev_w = None  

    def step(self, X, y, alpha, beta):
        """
        Compute one step of the gradient descent update using the feature matrix X
        and target vector y.
        The update rule is:
        w_new = w_old - alpha * grad + beta * (w_old - w_prev)
        where alpha is the learning rate, beta is the momentum coefficient,
        grad is the gradient of the loss function, and w_prev is the previous weight vector.
        """
        grad = self.model.grad(X, y)

        if self.prev_w is None:
            self.prev_w = self.model.w.clone()

        momentum = beta * (self.model.w - self.prev_w)

        new_w = self.model.w - alpha * grad + momentum

        self.prev_w = self.model.w.clone()
        self.model.w = new_w




class EvolutionOptimizer():
    """
    Evolutionary algorithm optimizer for the logistic regression model.
    This optimizer uses a population of individuals (weight vectors) and evolves them over generations.
    """
    def __init__(self, model):
        self.model = model
        self.mutation = 0.1  
        self.population_size = 100

    def set_mutation(self, mutation):
        """
        Set the mutation rate for the evolutionary algorithm.
        The mutation rate determines the probability of mutating each gene in the offspring.
        A higher mutation rate increases the diversity of the population but may also disrupt good solutions.
        """
        self.mutation = mutation
    
    def set_population_size(self, population_size):
        """
        Set the population size for the evolutionary algorithm.
        The population size determines the number of individuals in the population.
        """
        self.population_size = population_size

    def step(self, X, y):
        """
        Perform one step of the evolutionary algorithm.
        The algorithm works as follows:
        1. If the population is empty, initialize it with random weights.
        2. Sort the population based on their fitness (loss).
        3. Select the best half of the population.
        4. Create new individuals by combining genes from the best individuals.
        5. Mutate the new individuals.
        6. Update the population with the new individuals.
        7. Set the model's weights to the best individual in the population.
        """
        if self.model.population == []:
            self.model.population = [torch.rand(X.size(1)) for _ in range(self.population_size)]

        population = self.model.population
        population = sorted(population, key=lambda w: self.model.loss(X, y, w))
        population = population[:self.population_size // 2]  # Keep the best half

        new_population = []

        for _ in range(self.population_size): 
            parent1 = random.choice(population)
            parent2 = random.choice(population)

            # Inherit half from each parent
            mask = torch.rand_like(parent1) < 0.5
            child = torch.where(mask, parent1, parent2)

            mutation_mask = torch.rand_like(child) < self.mutation
            mutation_values = torch.normal(mean=0.0, std=0.1, size=child.size())
            child = torch.where(mutation_mask, child + mutation_values, child)

            new_population.append(child)

        # Update population and set model weights to best
        self.model.population = new_population
        best = min(new_population, key=lambda w: self.model.loss(X, y, w))
        self.model.w = best



    