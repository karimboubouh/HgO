import json
import os

import joblib
import numpy as np
import pandas as pd
from numpy import genfromtxt
from pandas import CategoricalDtype
from sklearn import preprocessing, datasets
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder, normalize, StandardScaler
from sklearn.utils import shuffle

from src.utils import divide_data, Map, mnist_noniid


def get_dataset(args):
    if args.dataset == 'wine':
        X_train, Y_train, X_test, Y_test = wine()
        train = Map({'data': X_train, 'targets': Y_train})
        test = Map({'data': X_test, 'targets': Y_test})
        masks = divide_data(X_train, args.workers)
    elif args.dataset == 'boston':
        X_train, Y_train, X_test, Y_test = boston()
        train = Map({'data': X_train, 'targets': Y_train})
        test = Map({'data': X_test, 'targets': Y_test})
        masks = divide_data(X_train, args.workers)
    elif args.dataset == 'adult':
        X_train, Y_train, X_test, Y_test = adult()
        train = Map({'data': X_train, 'targets': Y_train})
        test = Map({'data': X_test, 'targets': Y_test})
        masks = divide_data(X_train, args.workers)
    elif args.dataset == 'phishing':
        X_train, Y_train, X_test, Y_test = phishing()
        train = Map({'data': X_train, 'targets': Y_train})
        test = Map({'data': X_test, 'targets': Y_test})
        masks = divide_data(X_train, args.workers)
    elif args.dataset == 'msd':
        X_train, Y_train, X_test, Y_test = msd()
        train = Map({'data': X_train, 'targets': Y_train})
        test = Map({'data': X_test, 'targets': Y_test})
        masks = divide_data(X_train, args.workers)
    elif args.dataset == 'femnist':
        """
        Client: data: (80, 784), targets: (80,)
        Train : data: (24537, 784), targets: (24537,)
        Test  : data: (6237, 784), targets: (6237,)
        """
        if args.workers > 206:
            exit("!! Current femnist implementation supports at max {206} workers/writers.")
        clients, train, test = femnist()

        masks = {c: {'train': Map({'data': np.array(train[c]['x']), 'targets': encode(train[c]['y'])}),
                     'test': Map({'data': np.array(test[c]['x']), 'targets': encode(test[c]['y'])})
                     } for c in clients}
        train_X, train_Y, test_X, test_Y = [], [], [], []
        for c in clients:
            train_X.extend(train[c]['x'])
            train_Y.extend(train[c]['y'])
            test_X.extend(test[c]['x'])
            test_Y.extend(test[c]['y'])
        train = Map({'data': np.array(train_X), 'targets': encode(train_Y)})
        test = Map({'data': np.array(test_X), 'targets': encode(test_Y)})
        # for j in random.sample(range(1, 206), 20):
        #     print(test.targets[j])
        #     gen_image(test.data[j]).show()
        # exit()
    elif args.dataset == 'mnist':
        binary = False if args.model in ['DNN', 'CNN'] else True
        X_train, Y_train, X_test, Y_test = mnist(binary=binary)
        train = Map({'data': X_train, 'targets': Y_train})
        test = Map({'data': X_test, 'targets': Y_test})
        if args.iid == 1:
            masks = divide_data(X_train, args.workers)
        else:
            masks = mnist_noniid(train, args.workers, degree=args.iid_degree)
    elif args.dataset == 'cifar10':
        binary = False if args.model in ['DNN', 'CNN'] else True
        X_train, Y_train, X_test, Y_test = cifar10(binary=binary)
        train = Map({'data': X_train, 'targets': Y_train})
        test = Map({'data': X_test, 'targets': Y_test})
        masks = divide_data(X_train, args.workers)
    else:
        raise NotImplementedError(f"dataset {args.dataset} is not supported.")

    return train, test, masks


def encode(targets):
    # TODO activate encode
    return targets
    # return np.array([np.eye(1, 62, k=int(y)).reshape(62) for y in targets])


def get_wine_binary():
    data = genfromtxt("./data/wine.data", delimiter=",")
    np.random.shuffle(data)
    X, y = data[:, 1:], data[:, 0]
    # transform problem into binary classification task
    idxs = [i for i in range(len(y)) if y[i] == 1 or y[i] == 2]
    X, y = X[idxs], y[idxs]
    # Normalize data
    X = normalize(X)
    # Add bias to first column
    b = np.ones((X.shape[0], X.shape[1] + 1))
    b[:, 1:] = X
    X = b
    y = y.astype("int").reshape((-1, 1)) - 1

    return X, y


def wine():
    data = np.loadtxt('./datasets/wine.data', delimiter=',')
    X, y = data[:, 1:], data[:, 0]
    # transform problem into binary classification task
    idxs = [i for i in range(len(y)) if y[i] == 1 or y[i] == 2]
    X, y = X[idxs], y[idxs]
    # shuffle data
    X, y = shuffle(X, y)
    # normalize data
    X = (X - X.mean(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X = np.hstack((X, np.ones(len(X)).reshape(len(X), 1)))
    # transform target variable
    y = np.array(list(map(lambda x: 0 if x == 1 else 1, y)))
    y = y.reshape(-1, 1)

    return X, y, X, y


def load_adult(path="'./data/adult.csv'"):
    # load the dataset as a numpy array
    data = pd.read_csv(path, header=None, na_values='?')
    # drop rows with missing
    data = data.dropna()
    # split into inputs and outputs
    last_ix = len(data.columns) - 1
    X, y = data.drop(last_ix, axis=1), data[last_ix]
    # select categorical and numerical features
    cat_ix = X.select_dtypes(include=['object', 'bool']).columns
    num_ix = X.select_dtypes(include=['int64', 'float64']).columns
    # label encode the target variable to have the classes 0 and 1
    y = LabelEncoder().fit_transform(y)
    return X.values, y, cat_ix, num_ix


def get_adult(path="'./data/adult.csv'"):
    data = pd.read_csv(path, na_values='?')
    print(data.head())
    exit()
    data.dropna()
    # split into data and targets
    last_ix = len(data.columns) - 1
    X, y = data.drop(last_ix, axis=1), data[last_ix]
    target = data.values[:, -1]
    # select categorical and numerical features
    cat_ix = X.select_dtypes(include=['object', 'bool']).columns
    num_ix = X.select_dtypes(include=['int64', 'float64']).columns
    # label encode the target variable to have the classes 0 and 1
    y = LabelEncoder().fit_transform(y)
    return X.values, y, cat_ix, num_ix

    # counter = Counter(target)
    # for k, v in counter.items():
    #     per = v / len(target) * 100
    #     print('Class=%s, Count=%d, Percentage=%.3f%%' % (k, v, per))
    #
    # # data.info()
    # exit(0)
    # X, y = data[:, 1:], data[:, 0]
    #
    # # transform problem into binary classification task
    # idxs = [i for i in range(len(y)) if y[i] == 1 or y[i] == 2]
    #
    # X, y = X[idxs], y[idxs]
    #
    # # shuffle data
    # X, y = shuffle(X, y)
    #
    # # normalize data
    # X = (X - X.mean(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    # X = np.hstack((X, np.ones(len(X)).reshape(len(X), 1)))
    #
    # # transform target variable
    # y = np.array(list(map(lambda x: 0 if x == 1 else 1, y)))
    #
    # return X, y


def adult(path='./datasets/adult/'):
    """Preprocessing code fetched from
    https://github.com/animesh-agarwal/Machine-Learning-Datasets/tree/master/census-data"""
    train_data_path = os.path.join(path, 'adult.data')
    test_data_path = os.path.join(path, 'adult.test')
    try:
        open(train_data_path, 'r')
        open(test_data_path, 'r')
    except FileNotFoundError as e:
        print(str(e))
        print("Download `adult.data` and `adult.test` from "
              "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/")
        return None, None, None, None

    columns = ["age", "workClass", "fnlwgt", "education", "education-num",
               "marital-status", "occupation", "relationship", "race", "sex",
               "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]

    train_data = pd.read_csv(train_data_path, names=columns, engine='python', sep=' *, *', na_values='?')
    test_data = pd.read_csv(test_data_path, names=columns, engine='python', sep=' *, *', skiprows=1, na_values='?')

    num_attributes = train_data.select_dtypes(include=['int'])
    cat_attributes = train_data.select_dtypes(include=['object'])

    class ColumnsSelector(BaseEstimator, TransformerMixin):

        def __init__(self, type):
            self.type = type

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X.select_dtypes(include=[self.type])

    class CategoricalImputer(BaseEstimator, TransformerMixin):

        def __init__(self, columns=None, strategy='most_frequent'):
            self.columns = columns
            self.strategy = strategy

        def fit(self, X, y=None):
            if self.columns is None:
                self.columns = X.columns

            if self.strategy == 'most_frequent':
                self.fill = {column: X[column].value_counts().index[0] for column in self.columns}
            else:
                self.fill = {column: '0' for column in self.columns}

            return self

        def transform(self, X):
            X_copy = X.copy()
            for column in self.columns:
                X_copy[column] = X_copy[column].fillna(self.fill[column])
            return X_copy

    class CategoricalEncoder(BaseEstimator, TransformerMixin):

        def __init__(self, dropFirst=True):
            self.categories = dict()
            self.dropFirst = dropFirst

        def fit(self, X, y=None):
            join_df = pd.concat([train_data, test_data])
            join_df = join_df.select_dtypes(include=['object'])
            for column in join_df.columns:
                self.categories[column] = join_df[column].value_counts().index.tolist()
            return self

        def transform(self, X):
            X_copy = X.copy()
            X_copy = X_copy.select_dtypes(include=['object'])
            for column in X_copy.columns:
                X_copy[column] = X_copy[column].astype({column: CategoricalDtype(self.categories[column])})
            return pd.get_dummies(X_copy, drop_first=self.dropFirst)

    num_pipeline = Pipeline(steps=[
        ("num_attr_selector", ColumnsSelector(type='int')),
        ("scalar", StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ("cat_attr_selector", ColumnsSelector(type='object')),
        ("cat_imputer", CategoricalImputer(columns=['workClass', 'occupation', 'native-country'])),
        ("encoder", CategoricalEncoder(dropFirst=True))
    ])

    full_pipeline = FeatureUnion([("num_pipe", num_pipeline), ("cat_pipeline", cat_pipeline)])

    train_data.drop(['fnlwgt', 'education'], axis=1, inplace=True)
    test_data.drop(['fnlwgt', 'education'], axis=1, inplace=True)

    train_copy = train_data.copy()

    # convert the income column to 0 or 1 and then drop the column for the feature vectors
    train_copy["income"] = train_copy["income"].apply(lambda x: 0 if x == '<=50K' else 1)

    X_train = train_copy.drop('income', axis=1)
    Y_train = train_copy['income']

    X_train = full_pipeline.fit_transform(X_train)
    test_copy = test_data.copy()

    # convert the income column to 0 or 1
    test_copy["income"] = test_copy["income"].apply(lambda x: 0 if x == '<=50K.' else 1)

    # separating the feature vectors and the target values
    X_test = test_copy.drop('income', axis=1)
    Y_test = test_copy['income']

    # preprocess the test data using the full pipeline
    # here we set the type_df param to 'test'
    X_test = full_pipeline.fit_transform(X_test)

    s = preprocessing.MaxAbsScaler()
    X_train = s.fit_transform(X_train)
    X_test = s.transform(X_test)

    X_train = preprocessing.normalize(X_train)
    X_test = preprocessing.normalize(X_test)

    Y_train = Y_train.to_numpy()
    Y_test = Y_test.to_numpy()

    # Y_train[Y_train == 0] = -1
    # Y_test[Y_test == 0] = -1

    Y_train = Y_train.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)

    return X_train, Y_train, X_test, Y_test


def phishing(path='./datasets/phishing/'):
    data_path = os.path.join(path, 'phishing')
    try:
        open(data_path, 'r')
    except FileNotFoundError as e:
        print(str(e))
        print("Download phishing from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/phishing")
        return None, None, None, None

    X, Y = datasets.load_svmlight_file(data_path)

    X = X.toarray()
    Y[Y == 0] = -1

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    s = preprocessing.MaxAbsScaler()
    X_train = s.fit_transform(X_train)
    X_test = s.transform(X_test)

    X_train = preprocessing.normalize(X_train)
    X_test = preprocessing.normalize(X_test)

    Y_train = Y_train.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)

    return X_train, Y_train, X_test, Y_test


def boston(path=None):
    data = load_boston()
    X, Y = data.data, data.target
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    X = StandardScaler().fit_transform(X)  # for easy convergence
    X = np.hstack([X, np.ones((X.shape[0], 1))])

    Y = Y.reshape(-1, 1)
    #
    # #
    # # s = preprocessing.MaxAbsScaler()
    # # X_train = s.fit_transform(X_train)
    # # X_test = s.transform(X_test)
    # #
    # X_train = preprocessing.normalize(X_train)
    # X_test = preprocessing.normalize(X_test)
    #
    # Y_train = Y_train.reshape(-1, 1)
    # Y_test = Y_test.reshape(-1, 1)

    # return X_train, Y_train, X_test, Y_test
    return X, Y, X, Y


def msd(path='./datasets/MSD/'):
    train_data_path = os.path.join(path, 'YearPredictionMSD.data')
    test_data_path = os.path.join(path, 'YearPredictionMSD.test')
    try:
        open(train_data_path, 'r')
        open(test_data_path, 'r')
    except FileNotFoundError as e:
        print(str(e))
        print("Download `YearPredictionMSD.data` and `YearPredictionMSD.test` from "
              "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/")
        exit()

    X_train, Y_train = datasets.load_svmlight_file(train_data_path)
    X_test, Y_test = datasets.load_svmlight_file(test_data_path)

    X_train = X_train.toarray()
    X_test = X_test.toarray()

    y_m = np.mean(Y_train)
    y_s = np.std(Y_train)

    Y_train = (Y_train - y_m)
    Y_test = (Y_test - y_m)

    s = preprocessing.MaxAbsScaler()
    X_train = s.fit_transform(X_train)
    X_test = s.transform(X_test)

    X_train = preprocessing.normalize(X_train)
    X_test = preprocessing.normalize(X_test)

    return X_train, Y_train, X_test, Y_test


def mnist(path='./datasets/mnist/', train_size=60000, binary=True):
    data_path = os.path.join(path, 'mnist.data')
    try:
        open(data_path, 'r')
    except FileNotFoundError as e:
        print(str(e))
        print("Download <mnist.data> from http://tiny.cc/mnist")
        exit()

    X, Y = joblib.load(data_path)

    X_train = X[:train_size]
    Y_train = Y[:train_size].astype(int).reshape(-1, 1)
    X_test = X[train_size:]
    Y_test = Y[train_size:].astype(int).reshape(-1, 1)

    if binary:
        # Extract 1 and 2 from train dataset
        f1 = 1
        f2 = 2
        Y_train = np.squeeze(Y_train)
        X_train = X_train[np.any([Y_train == f1, Y_train == f2], axis=0)]
        Y_train = Y_train[np.any([Y_train == f1, Y_train == f2], axis=0)]
        Y_train = Y_train - f1
        Y_train = Y_train.reshape(-1, 1)

        # Extract 1 and 2 from train dataset
        Y_test = np.squeeze(Y_test)
        X_test = X_test[np.any([Y_test == f1, Y_test == f2], axis=0)]
        Y_test = Y_test[np.any([Y_test == f1, Y_test == f2], axis=0)]

        Y_test = Y_test - f1
        Y_test = Y_test.reshape(-1, 1)
    else:
        Y_train = np.array([np.eye(1, 10, k=int(y)).reshape(10) for y in Y_train])
        Y_test = np.array([np.eye(1, 10, k=int(y)).reshape(10) for y in Y_test])
        # pass
        # X_train, X_test = X_train.T, X_test.T
        # Y_train, Y_test = Y_train.T, Y_test.T

    # Normalize data
    X_train = X_train / 255
    X_test = X_test / 255

    return X_train, Y_train, X_test, Y_test


def cifar10(path='./datasets/cifar/', binary=True):
    data_path = os.path.join(path, 'cifar10.data')
    try:
        open(data_path, 'r')
    except FileNotFoundError as e:
        print(str(e))
        print("Download cifar10.data from http://tiny.cc/cifar10")
        return None, None, None, None

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    X_train, Y_train, X_test, Y_test = joblib.load(data_path)
    X_train = np.reshape(X_train, (X_train.shape[0], -1))  # [50000, 3072]
    X_test = np.reshape(X_test, (X_test.shape[0], -1))  # [10000, 3072]
    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image
    if binary:
        c1 = classes.index('plane')
        c2 = classes.index('car')
        idxs = np.logical_or(Y_train == c1, Y_train == c2)
        X_train = X_train[idxs, :]
        Y_train = Y_train[idxs]
        idxs = np.logical_or(Y_test == c1, Y_test == c2)
        X_test = X_test[idxs, :]
        Y_test = Y_test[idxs]
        Y_train = Y_train.reshape(-1, 1)
        Y_test = Y_test.reshape(-1, 1)
    else:
        pass
        # Y_train = np.array([np.eye(1, 10, k=int(y)).reshape(10) for y in Y_train])
        # Y_test = np.array([np.eye(1, 10, k=int(y)).reshape(10) for y in Y_test])
        # X_train, X_test = X_train.T, X_test.T
        # Y_train, Y_test = Y_train.T, Y_test.T

    return X_train, Y_train, X_test, Y_test


def femnist(path='./datasets/femnist/'):
    print(">> Loading femnist dataset ...")
    # train file
    train_file = os.path.join(path, "train.json")
    with open(train_file, 'r') as inf:
        cdata = json.load(inf)
        train_data = {int(k): v for k, v in cdata['user_data'].items()}

    # test file
    test_file = os.path.join(path, "test.json")
    with open(test_file, 'r') as inf:
        cdata = json.load(inf)
        test_data = {int(k): v for k, v in cdata['user_data'].items()}

    # client ids
    clients = list(sorted(train_data.keys()))

    return clients, train_data, test_data


if __name__ == '__main__':
    clients_, train_data_, test_data_ = femnist(path='../datasets/femnist/')

    for i in clients_:
        print(f"client {i} has {train_data_[i]['y']} data samples\n\n\n")
