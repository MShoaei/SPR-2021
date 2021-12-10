import numpy as np


class GLDA(object):
    """
    GLDA is the class for Gaussian Learning Discriminant Analysis
    """

    def __init__(self):
        """
        Initialize GLDA
        """
        self.mu_0 = None
        self.mu_1 = None
        self.Sigma_0 = None
        self.Sigma_1 = None
        self.prior_0 = None
        self.prior_1 = None
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Fit the model according to the given training data
        """
        # Check if the training data is valid
        if len(X) != len(y):
            raise ValueError(
                "The number of training data and labels are not equal")
        if len(np.unique(y)) != 2:
            raise ValueError("The number of classes is not equal to 2")

        # Initialize the parameters
        self.mu_0 = np.mean(X[y == 0], axis=0)
        self.mu_1 = np.mean(X[y == 1], axis=0)
        self.Sigma_0 = np.cov(X[y == 0].T)
        self.Sigma_1 = np.cov(X[y == 1].T)
        self.prior_0 = len(X[y == 0]) / len(X)
        self.prior_1 = len(X[y == 1]) / len(X)

        # Calculate the weights and bias
        self.w = np.linalg.inv(self.Sigma_1).dot(self.mu_1 - self.mu_0)
        self.b = -0.5 * self.w.T.dot(self.mu_0) + \
            np.log(self.prior_0 / self.prior_1)

    def predict(self, X):
        """
        Predict the class labels for the given test data
        """
        # Check if the model has been fitted
        if self.w is None:
            raise NotImplementedError("The model has not been fitted yet")

        # Calculate the class labels
        y_pred = np.zeros(len(X))
        for i, x in enumerate(X):
            y_pred[i] = 1 if self.w.T.dot(x) + self.b > 0 else 0

        return y_pred

    def score(self, X, y):
        """
        Calculate the accuracy of the model
        """
        # Check if the model has been fitted
        if self.w is None:
            raise NotImplementedError("The model has not been fitted yet")

        # Calculate the accuracy
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
