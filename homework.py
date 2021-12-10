import click
from sklearn import datasets, model_selection
import pandas as pd

import preprocess
from scripts import *


@click.command()
@click.option("--normalize", type=click.Choice(["zero-mean", "min-max"]), default="zero-mean", help="Normalization algorithm to use on the data")
@click.option("--algo", type=click.Choice(["lstsq", "bgd", "sgd", "all"]), help="learning algorithm")
@click.option("--train", type=click.Path(exists=True), help="Path to the training data")
@click.option("--test", type=click.Path(exists=True), help="Path to the test data")
def hw1(normalize, algo, train, test):
    df_train = pd.read_csv(train)
    df_train.insert(0, "bias", 1)

    df_test = pd.read_csv(test)

    norm = preprocess.zero_mean_normalize if normalize == "zero-mean" else preprocess.min_max_normalize
    norm(df_train, ["x"])
    norm(df_test, ["x"])

    if algo == "lstsq":
        lstsq(df_train, df_test)
    elif algo == "bgd":
        bgd(df_train, df_test)
    elif algo == "sgd":
        sgd(df_train, df_test)
    elif algo == "all":
        lstsq(df_train, df_test)
        bgd(df_train, df_test)
        sgd(df_train, df_test)


@click.command()
@click.option('--normalize', type=click.Choice(['zero-mean', 'min-max']), default='zero-mean', help='Normalization algorithm to use on the data')
@click.option('--algo', type=click.Choice(['bi-logistic', 'multi-logistic', 'softmax', 'all']), help='learning algorithm')
def hw2(normalize, algo):
    data = datasets.load_iris(as_frame=True).frame

    df_train, df_test = model_selection.train_test_split(
        data, train_size=0.8, shuffle=True, random_state=1)

    norm = preprocess.zero_mean_normalize if normalize == 'zero-mean' else preprocess.min_max_normalize
    norm(df_train, df_train.columns[:-1])
    norm(df_test, df_test.columns[:-1])

    if algo == 'bi-logistic':
        train = df_train.loc[df_train['target'] != 1].copy()
        train.loc[:, 'target'] = df_train['target'].apply(
            lambda x: 1 if x > 0 else 0)

        test = df_test[df_test['target'] != 1].copy()
        test.loc[:, 'target'] = df_test['target'].apply(
            lambda x: 1 if x > 0 else 0)

        bi_logistic(train.iloc[:, [0, 1, -1]], test.iloc[:, [0, 1, -1]])
    elif algo == 'multi-logistic':
        multi_logistic(df_train, df_test)
    elif algo == 'softmax':
        softmax(df_train, df_test)
    elif algo == 'all':
        train = df_train.loc[df_train['target'] != 1].copy()
        train.loc[:, 'target'] = df_train['target'].apply(
            lambda x: 1 if x > 0 else 0)

        test = df_test[df_test['target'] != 1].copy()
        test.loc[:, 'target'] = df_test['target'].apply(
            lambda x: 1 if x > 0 else 0)

        bi_logistic(train.iloc[:, [0, 1, -1]], test.iloc[:, [0, 1, -1]])
        multi_logistic(df_train, df_test)
        softmax(df_train, df_test)


@click.command()
@click.option('--normalize', type=click.Choice(['zero-mean', 'min-max']), default='zero-mean', help='Normalization algorithm to use on the data')
@click.option('--algo', type=click.Choice(['bayesian', 'quadratic-multi', 'naive-bayes', 'all']), help='learning algorithm')
def hw3(normalize, algo):
    df_train = pd.read_csv(
        'HW3/BC-Test1.csv', names=['x1', 'x2', 'label']).to_numpy()

    norm = preprocess.zero_mean_normalize if normalize == 'zero-mean' else preprocess.min_max_normalize
    norm(df_train, ['x1', 'x2'])

    if algo == 'bayesian':
        pass
    elif algo == 'quadratic-multi':
        pass
    elif algo == 'naive-bayes':
        pass
    elif algo == 'all':
        pass
