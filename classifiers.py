import numpy as np
import pandas as pd
from metrics import cross_entropy_error
from transform import one_hot_encoding


class BinaryClassifier(object):
    def __init__(self, eta=1e-4, tol=1e-3, n_iters=1e2):
        self.eta = eta
        self.tol = tol
        self.n_iters = n_iters

        self.theta_ = None
        self.coef_ = None
        self.costs_ = None

    def fit(self, X, y):
        assert X.shape[0] == y.shape[0], \
            "the size of X must be equal to the size of y"
        y = y.to_numpy().reshape(-1, 1)

        self.theta_ = np.zeros((X.shape[1], 1))
        self.coef_ = self.theta_[1:]
        self.costs_ = np.array([])

        for i in range(int(self.n_iters)):
            self.costs_ = np.append(self.costs_, self.cost(X, y))
            if (len(self.costs_) > 1) and abs(self.costs_[-1]-self.costs_[-2]) < self.tol:
                break
            gradients = np.matmul(X.T, self.sigmoid(
                np.matmul(X, self.theta_))-y)
            self.theta_ -= self.eta * gradients

        self.intercept_ = self.theta_[0, 0]
        return self

    def predict(self, X):
        assert self.theta_ is not None, \
            "must fit before predict"

        return np.where(self.sigmoid(np.matmul(X, self.coef_)+self.intercept_) >= 0.5, 1, 0)

    def predict_proba(self, X):
        return self.sigmoid(np.matmul(X, self.theta_))

    def cost(self, X, y):
        y_pred = self.predict_proba(X)
        return -np.mean(np.matmul(y.T, np.log(y_pred)) + np.matmul((1 - y).T, np.log(1 - y_pred)))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


class MulticlassClassifier(object):
    def __init__(self, eta=1e-4, tol=1e-4, n_iters=1e2, solver='ovo'):
        if solver not in ['ovo', 'ovr']:
            raise ValueError("solver must be 'ovo' or 'ovr'")
        self.solver = solver
        self.eta = eta
        self.tol = tol
        self.n_iters = int(n_iters)

        self.theta_ = None
        self.coef_ = None
        self.costs_ = np.array([])
        self.n_classes_ = None
        self.classes_ = None
        self.convergence_ = None

    def fit_logistic_(self, X, y):
        theta = np.zeros((X.shape[1], 1))
        y = y.reshape(-1, 1)
        old_cost = -np.inf
        for i in range(self.n_iters):
            y_pred = self.sigmoid(np.matmul(X, theta))
            new_cost = -np.mean(np.matmul(y.T, np.log(y_pred)) +
                                np.matmul((1 - y).T, np.log(1 - y_pred)))
            if (i > 1) and new_cost-old_cost > 0:
                break

            gradients = np.matmul(X.T, self.sigmoid(np.matmul(X, theta)) - y)
            theta -= self.eta * gradients
            old_cost = new_cost
        return theta

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        y_ = None

        if self.solver == 'ovo':
            self.theta_ = np.zeros(
                (X.shape[1], self.n_classes_*(self.n_classes_-1)//2))

            self.pairs = [(i, j)
                          for i in self.classes_ for j in self.classes_ if i < j]
            for p in self.pairs:
                y_ = y[(y == p[0]) | (y == p[1])].copy()
                y_ = np.where(y_ == p[0], 1, 0)

                X_ = X[(y == p[0]) | (y == p[1])].copy()

                theta = self.fit_logistic_(X_, y_)
                self.theta_[:, self.pairs.index(p)] = theta.reshape(-1)

        elif self.solver == 'ovr':
            self.theta_ = np.zeros((X.shape[1], self.n_classes_))

            self.coef_ = self.theta_[:, 1:]
            self.intercept_ = self.theta_[:, 0]
            for i in range(self.n_iters):
                self.costs_ = np.append(
                    self.costs_, cross_entropy_error(one_hot_encoding(y.to_numpy(), self.n_classes_), self.predict_proba(X)))
                if (len(self.costs_) > 1) and abs(self.costs_[-1]-self.costs_[-2]) < self.tol:
                    self.convergence_ = i-1
                    self.costs_ = self.costs_[:-1]
                    break
                for j in range(self.n_classes_):
                    y_ = y.copy()
                    y_[y != j] = 0
                    y_[y == j] = 1

                    y_pred = self.sigmoid(np.matmul(X, self.theta_[:, j]))
                    gradients = np.matmul(X.T, y_pred-y_)
                    self.theta_[:, j] -= self.eta * gradients
            self.intercept_ = self.theta_[:, 0]

        return self

    def predict_proba(self, X):
        assert self.theta_ is not None, "Must fit before predict"
        assert self.theta_.shape[0] == X.shape[1], "X must have same number of features as theta"

        return self.sigmoid(np.matmul(X, self.theta_))

    def predict(self, X):
        assert self.theta_ is not None, "Must fit before predict"
        assert self.theta_.shape[0] == X.shape[1], "X must have same number of features as theta"
        result = np.zeros((X.shape[0], 1))
        if self.solver == 'ovo':
            y_pred = self.predict_proba(X)
            for i, p in enumerate(self.pairs):
                y_pred[:, i] = np.where(y_pred[:, i] >= 0.5, p[0], p[1])

            for i, row in enumerate(y_pred):
                labels, counts = np.unique(row, return_counts=True)
                # print(np.argmax(counts),labels[np.argmax(counts)])
                result[i] = labels[np.argmax(counts)]
            return result
        elif self.solver == 'ovr':
            return np.argmax(self.predict_proba(X), axis=1)

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def cost(self, X, y):
        y_pred = self.predict_proba(X)
        return -np.mean(np.matmul(y.T, np.log(y_pred)) + np.matmul((1 - y).T, np.log(1 - y_pred)))


class Softmax(object):
    def __init__(self, eta=1e-4, tol=1e-4, n_iters=1e2):
        self.eta = eta
        self.tol = tol
        self.n_iters = int(n_iters)

        self.n_classes_ = None
        self.classes_ = None

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.classes_ = np.unique(y)

        self.theta_ = np.zeros((self.n_classes_, X.shape[1]))

        y = one_hot_encoding(y.to_numpy(), self.n_classes_)
        for _ in range(self.n_iters):
            y_pred = self.predict_proba(X)
            grad = np.matmul((y_pred - y).T, X)
            self.theta_ -= self.eta * grad

    def predict_proba(self, X):
        assert self.theta_ is not None, "Must fit before predict"
        assert self.theta_.shape[1] == X.shape[1], "X must have same number of features as theta"

        z = np.matmul(X, self.theta_.T)
        return self.softmax(z)

    def predict(self, X):
        assert self.theta_ is not None, "Must fit before predict"
        assert self.theta_.shape[1] == X.shape[1], "X must have same number of features as theta"

        y_pred = self.predict_proba(X)
        return np.argmax(y_pred, axis=1)

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(z.shape[0], 1)
