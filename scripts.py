import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import regressors
from metrics import *
import classifiers


# HW1 START


def lstsq(train: pd.DataFrame, test: pd.DataFrame):
    reg = regressors.LeastSquareRegressor()
    reg.fit(train[['bias', 'x']], train['y'])
    print('theta:', reg.theta_)
    print('coefs:', reg.coef_)
    print('y-intercept:', reg.intercept_)
    mse_train = mean_squared_error(
        train['y'], reg.predict(train[['x']]))
    print('MSE (train):', mse_train)

    mse_test = mean_squared_error(
        test['y'], reg.predict(test[['x']]))
    print('MSE (test):', mse_test)

    plt.clf()
    plt.scatter(train[['x']], train['y'], s=1, label='train data')
    plt.plot(train[['x']], reg.predict(train[['x']]),
             c='red', label='fitted line')
    plt.legend()
    plt.savefig('HW1/report/lstsq-train.png')

    plt.clf()
    plt.scatter(test[["x"]], test["y"], s=1, label="test data")
    plt.plot(test[["x"]], reg.predict(test[["x"]]),
             lw=1, c="red", label="fitted line")
    plt.legend()
    plt.savefig("HW1/report/lstsq-test.png")


def bgd(train: pd.DataFrame, test: pd.DataFrame):
    reg = regressors.BGDRegressor()
    reg.fit(train[["bias", "x"]], train["y"])
    print("theta:", list(reg.theta_))
    print("coefs:", reg.coef_)
    print("y-intercept:", reg.intercept_)

    mse_train = mean_squared_error(
        train["y"], reg.predict(train[["x"]]))
    print("MSE (train):", mse_train)

    mse_test = mean_squared_error(
        test["y"], reg.predict(test[["x"]]))
    print("MSE (test):", mse_test)

    plt.clf()
    plt.scatter(train[["x"]], train["y"], s=1, label="train data")
    plt.plot(train[["x"]], reg.predict(train[["x"]]),
             c="red", label="fitted line")
    plt.legend()
    plt.savefig("HW1/report/bgd-train.png")

    plt.clf()
    plt.scatter(list(range(len(reg.costs_))), reg.costs_)
    plt.plot(list(range(len(reg.costs_))), reg.costs_, label="costs")
    plt.legend()
    plt.savefig("HW1/report/bgd-costs.png")

    plt.clf()
    plt.scatter(test[["x"]], test["y"], s=1, label="test data")
    plt.plot(test[["x"]], reg.predict(test[["x"]]),
             lw=1, c="red", label="fitted line")
    plt.legend()
    plt.savefig("HW1/report/bgd-test.png")


def sgd(train: pd.DataFrame, test: pd.DataFrame):
    reg = regressors.SGDRegressor()
    reg.fit(train[["bias", "x"]], train["y"])
    print("theta:", list(reg.theta_))
    print("coefs:", reg.coef_)
    print("y-intercept:", reg.intercept_)

    mse_train = mean_squared_error(
        train["y"], reg.predict(train[["x"]]))
    print("MSE (train):", mse_train)

    mse_test = mean_squared_error(
        test["y"], reg.predict(test[["x"]]))
    print("MSE (test):", mse_test)

    plt.clf()
    plt.scatter(train[["x"]], train["y"], s=1, label="train data")
    plt.plot(train[["x"]], reg.predict(train[["x"]]),
             lw=1, c="red", label="fitted line")
    plt.legend()
    plt.savefig("HW1/report/sgd-train.png")

    plt.clf()
    plt.scatter(list(range(len(reg.costs_))), reg.costs_,)
    plt.plot(list(range(len(reg.costs_))), reg.costs_, label="costs")
    plt.legend()
    plt.savefig("HW1/report/sgd-costs.png")

    plt.clf()
    plt.scatter(test[["x"]], test["y"], s=1, label="test data")
    plt.plot(test[["x"]], reg.predict(test[["x"]]),
             lw=1, c="red", label="fitted line")
    plt.legend()
    plt.savefig("HW1/report/sgd-test.png")


# HW1 END
# HW2 START

def bi_logistic(train: pd.DataFrame, test: pd.DataFrame):
    train.insert(0, "bias", 1)
    test.insert(0, "bias", 1)

    reg = classifiers.BinaryClassifier(eta=1e-4, tol=1e-4, n_iters=1e5)
    reg.fit(train.iloc[:, :-1].to_numpy(), train.iloc[:, -1])
    print("theta:", reg.theta_.flatten())
    print("coefs:", reg.coef_.flatten())
    print("y-intercept:", reg.intercept_)

    train_error = cross_entropy_error(
        train.iloc[:, -1].to_numpy(), reg.predict_proba(train.iloc[:, [0, 1, 2]].to_numpy())[:, 0])
    print("cross-entropy (train):", train_error)

    test_error = cross_entropy_error(
        test.iloc[:, -1].to_numpy(), reg.predict_proba(test.iloc[:, [0, 1, 2]].to_numpy())[:, 0])
    print("cross-entropy (test):", test_error)

    slope = reg.theta_[1, 0]/reg.theta_[2, 0]
    intercept = reg.theta_[0, 0]/reg.theta_[2, 0]
    print("decision boundary equation:", f"y = {slope:.4f}x + {intercept:.4f}")

    xx, yy = np.meshgrid(np.linspace(-2.5, 3.5, 1000),
                         np.linspace(-3, 3.5, 1000))
    zz = reg.predict(
        np.c_[xx.ravel(),
              yy.ravel()]).reshape(xx.shape)
    plt.clf()
    plt.title("Binary classification cost")
    plt.scatter(range(0, len(reg.costs_), 100), reg.costs_[::100], marker='.')
    plt.xlabel("iteration")
    plt.ylabel("cost")
    plt.savefig('HW2/report/bi-logistic-costs.png')

    plt.clf()
    plt.title("Binary classification")
    plt.scatter(train.iloc[:, 1], train.iloc[:, 2], c=train['target'], s=100, cmap=plt.cm.Paired,
                edgecolors='black')
    plt.pcolormesh(xx, yy, zz, cmap=plt.cm.Paired, shading="auto", alpha=0.2)
    plt.xlabel(train.columns[1])
    plt.ylabel(train.columns[2])
    plt.savefig('HW2/report/bi-logistic-train.png')


def multi_logistic(train: pd.DataFrame, test: pd.DataFrame):
    train.insert(0, 'bias', 1)
    test.insert(0, "bias", 1)

    reg = classifiers.MulticlassClassifier(
        eta=1e-3, tol=1e-4, n_iters=1e4, solver='ovr')
    reg.fit(train.iloc[:, :-1].to_numpy(), train.iloc[:, -1])

    print("METHOD: One-vs-Rest (OvR)")
    print("theta:", reg.theta_)
    train_accuracy = accuracy_score(
        train['target'].to_numpy(), reg.predict(train.iloc[:, :-1].to_numpy()))
    print("accuracy (train):", train_accuracy)

    test_accuracy = accuracy_score(
        test['target'].to_numpy(), reg.predict(test.iloc[:, :-1].to_numpy()))
    print("accuracy (test):", test_accuracy)

    print("convergence iteration:", reg.convergence_)

    print()
    print()

    plt.clf()
    plt.title("Multiclass classification cost")
    plt.scatter(range(0, len(reg.costs_), 1), reg.costs_[::1], marker='.')
    plt.xlabel("iteration")
    plt.ylabel("cost")
    plt.savefig('HW2/report/multi-logistic-costs.png')

    reg = classifiers.MulticlassClassifier(
        eta=1e-4, n_iters=1e5, solver='ovo')
    reg.fit(train.iloc[:, :-1].to_numpy(), train.iloc[:, -1])

    print("METHOD: One-vs-One (OvO)")
    train_accuracy = accuracy_score(
        train['target'].to_numpy(), reg.predict(train.iloc[:, :-1].to_numpy()).flatten())
    print("accuracy (train):", train_accuracy)

    test_accuracy = accuracy_score(
        test['target'].to_numpy(), reg.predict(test.iloc[:, :-1].to_numpy()).flatten())
    print("accuracy (test):", test_accuracy)


def softmax(train: pd.DataFrame, test: pd.DataFrame):
    # TODO: implement softmax
    return

# HW2 END
# HW3 START


def bayesian(train: pd.DataFrame, test: pd.DataFrame):
    pass


def quadratic_multi(train: pd.DataFrame, test: pd.DataFrame):
    pass


def naive_bayes(train: pd.DataFrame, test: pd.DataFrame):
    pass

# HW3 END
