import numpy as np
import pandas as pd


class LeastSquareRegressor(object):
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.theta_ = None

    def fit(self, X, y):
        y = y.to_numpy().reshape(-1, 1)
        # self.theta_ = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.theta_ = np.matmul(np.matmul(np.linalg.inv(
            np.matmul(X.T, X)), X.T), y)
        self.intercept_ = self.theta_.iloc[0, 0]
        self.coef_ = self.theta_.iloc[1:, 0].to_numpy()

        return self

    def predict(self, X):
        assert self.coef_ is not None and self.intercept_ is not None, \
            "must fit before predict"
        assert X.shape[1] == len(self.coef_), \
            "the feature number of X must be equal to X_train"

        return X.dot(self.coef_)+self.intercept_


class BGDRegressor(object):
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.theta_ = None
        self.costs_ = None

    def fit(self, X, y, eta=0.0001, n_iters=1e2):
        assert X.shape[0] == y.shape[0], \
            "the size of X must be equal to the size of y"

        self.theta_ = np.zeros(X.shape[1])
        self.coef_ = self.theta_[1:]
        self.costs_ = np.array([])
        for i in range(int(n_iters)):
            self.costs_ = np.append(self.costs_, self.cost(X, y))
            gradients = X.T.dot(X.dot(self.theta_) - y)
            self.theta_ -= eta * gradients
        self.intercept_ = self.theta_[0]
        return self

    def predict(self, X):
        assert self.coef_ is not None and self.intercept_ is not None, \
            "must fit before predict"
        assert X.shape[1] == len(self.coef_), \
            "the feature number of X must be equal to X_train"

        return X.dot(self.coef_) + self.intercept_

    def cost(self, X, y):
        assert X.shape[0] == y.shape[0], \
            "the size of X must be equal to the size of y"

        return np.mean((X.dot(self.theta_) - y)**2)


class SGDRegressor(object):
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.theta_ = None
        self.costs_ = None

    def fit(self, X: pd.DataFrame, y, eta=0.01, n_iters=1):
        assert X.shape[0] == y.shape[0], \
            "the size of X must be equal to the size of y"

        X.sample(frac=1).reset_index(drop=True)

        self.theta_ = np.zeros(X.shape[1])
        self.coef_ = self.theta_[1:]
        self.costs_ = np.array([])
        for i in range(int(n_iters)):
            for j in range(X.shape[0]):
                self.costs_ = np.append(self.costs_, self.cost(X, y))
                gradients = X.iloc[[j]].T.dot(
                    X.iloc[[j]].dot(self.theta_) - y[j])
                self.theta_ -= eta * gradients
        self.intercept_ = self.theta_[0]

        return self

    def predict(self, X):
        assert self.coef_ is not None and self.intercept_ is not None, \
            "must fit before predict"
        assert X.shape[1] == len(self.coef_), \
            "the feature number of X must be equal to X_train"

        return X.dot(self.coef_) + self.intercept_

    def cost(self, X, y):
        assert X.shape[0] == y.shape[0], \
            "the size of X must be equal to the size of y"

        return np.mean((X.dot(self.theta_) - y)**2)
