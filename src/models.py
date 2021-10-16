import time

import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from sklearn.utils import shuffle

from src.datasets import phishing, mnist
from src.nonlinear_models import DNN
from src.optimizers import LROptimizer, LNOptimizer, SVMOptimizer
from src.utils import get_batch, sigmoid, accuracy, chunks


def load_model(n_features, args):
    if args.model == 'LR':
        model = LogisticRegression(n_features, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size)
    elif args.model == 'LN':
        model = LinearRegression(n_features, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size)
    elif args.model == 'RR':
        model = RidgeRegression(n_features, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size)
    elif args.model == 'SVM':
        model = SVM(n_features, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size)
    elif args.model == 'DNN':
        if isinstance(n_features, list):
            model = DNN(n_features, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size)
        else:
            raise AttributeError("Please provide a list of layer dimensions.")
    else:
        raise NotImplementedError()

    return model


class LogisticRegression(object):
    """
    Logistic Regression
    """

    def __init__(self, n_features, lr=0.001, epochs=200, batch_size=128, threshold=0.5, debug=True):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = None
        self.threshold = threshold
        self.debug = debug
        self.costs = []
        self._accuracies = []
        self.W = np.random.randn(n_features, 1) * 0.01

    def reset(self):
        self.optimizer = None
        self.costs = []
        self._accuracies = []
        self.W = np.random.randn(len(self.W), 1) * 0.01

    def fit(self, X, y, optimizer=LROptimizer):
        self.costs = []
        self._accuracies = []
        BLOCK_SIZE = 32
        blocks = chunks(list(range(len(self.W))), BLOCK_SIZE)
        self.optimizer = optimizer(self.W, self.lr)
        batches = (X.shape[0] // self.batch_size)
        for i in range(self.epochs + 1):
            X, y = shuffle(X, y)
            if blocks:
                index = np.random.choice(len(blocks), replace=False)
                self.optimizer.block = blocks[index]
            for j in range(batches):
                batch = get_batch(X, y, self.batch_size, j)
                features, labels = batch
                # Foreword step
                predictions = self.forward(features)
                # print(features.shape, predictions.shape)
                # Optimization step
                grad, _ = self.optimizer.optimize(labels, predictions, features)
                if blocks:
                    b = self.optimizer.block
                    self.W[b] = self.W[b] - self.lr * grad
                else:
                    self.W = self.W - self.lr * grad

            # Evaluation
            predictions = self.forward(X)
            cost = self.optimizer.loss(y, predictions)
            acc = accuracy(y, predictions)
            self.costs.append(cost)
            self._accuracies.append(acc)
            if i % 10 == 0 and self.debug:
                print("Epoch {}, Loss {}, Acc {}".format(i, round(cost, 4), round(acc, 4)))

    def one_epoch(self, X, y, block, si):
        self.optimizer.W = self.W
        self.optimizer.block = block
        self.batch_size = si
        features, labels = self.get_random_batch(X, y)
        # Foreword step
        predictions = self.forward(features)
        # Optimization step
        grads, gtime = self.optimizer.optimize(labels, predictions, features)

        # self.optimizer.estimate_lr_GD(features, labels, block)
        # self.optimizer.estimate_lr_CD(features, labels, block)

        return grads, gtime

    def get_random_batch(self, X, y):
        sX, sy = shuffle(X, y)
        m = X.shape[0]

        if m < self.batch_size:
            self.batch_size = m
        nb_batches = (m // self.batch_size)
        j = np.random.choice(nb_batches, replace=False)
        return get_batch(sX, sy, self.batch_size, j)

    def forward(self, X):
        # a = np.dot(self.W.T, X.T)
        a = X @ self.W
        return sigmoid(a)

    def predict(self, X):
        y_pred = sigmoid(np.dot(self.W.T, X.T))
        return np.array(list(map(lambda x: 1 if x >= 0.5 else 0, y_pred.flatten())))

    def evaluate(self, X, y):
        predictions = self.forward(X)
        cost = self.optimizer.loss(y, predictions)
        acc = accuracy(y, predictions)
        return cost, acc

    def summary(self, X, y):
        print("-------------------")
        print(f"Training for {self.epochs} epochs.")
        print(f">> Train:\n\tloss: {round(self.costs[-1], 4)}.")
        print(f"\taccuracy: {round(self._accuracies[-1], 3) * 100}%.")
        cost, acc = self.evaluate(X, y)
        print(f">> Test:\n\tloss: {round(cost, 4)}.")
        print(f"\taccuracy: {round(acc, 3) * 100}%.")
        return self


class MultiClassLogisticRegression:

    def __init__(self, n_iter=500, thres=1e-3, batch_size=64, lr=0.001, verbose=True):
        self.n_iter = n_iter
        self.thres = thres
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose
        self.loss = []
        self.classes = []
        self.class_labels = []
        self.weights = None

    def fit(self, X, y, rand_seed=4):
        np.random.seed(rand_seed)
        self.classes = np.unique(y)
        self.class_labels = {c: i for i, c in enumerate(self.classes)}
        X = self.add_bias(X)
        y = self.one_hot(y)
        self.loss = []
        self.weights = np.zeros(shape=(len(self.classes), X.shape[1]))
        self.fit_data(X, y)
        return self

    def fit_data(self, X, y):
        i = 0
        while not self.n_iter or i < self.n_iter:
            self.loss.append(self.cross_entropy(y, self.predict_(X)))
            idx = np.random.choice(X.shape[0], self.batch_size)
            X_batch, y_batch = X[idx], y[idx]
            error = y_batch - self.predict_(X_batch)
            update = (self.lr * np.dot(error.T, X_batch))
            self.weights += update
            if np.abs(update).max() < self.thres: break
            if i % 100 == 0 and self.verbose:
                print(' Training Accuracy at {} iterations is {}'.format(i, self.evaluate_(X, y)))
            i += 1

    def predict(self, X):
        return self.predict_(self.add_bias(X))

    def predict_(self, X):
        pre_vals = np.dot(X, self.weights.T).reshape(-1, len(self.classes))
        return self.softmax(pre_vals)

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1, 1)

    def predict_classes(self, X):
        self.probs_ = self.predict(X)
        return np.vectorize(lambda c: self.classes[c])(np.argmax(self.probs_, axis=1))

    def add_bias(self, X):
        return np.insert(X, 0, 1, axis=1)

    def get_randon_weights(self, row, col):
        return np.zeros(shape=(row, col))

    def one_hot(self, y):
        return np.eye(len(self.classes))[np.vectorize(lambda c: self.class_labels[c])(y).reshape(-1)]

    def score(self, X, y):
        return np.mean(np.argmax(self.predict_(X), axis=1) == np.argmax(y, axis=1))

    def evaluate_(self, X, y):
        return np.mean(np.argmax(self.predict_(X), axis=1) == np.argmax(y, axis=1))

    def cross_entropy(self, y, probs):
        return -1 * np.mean(y * np.log(probs))


class LinearRegression(object):
    """
    Linear Regression
    """

    def __init__(self, n_features, lr=0.01, epochs=200, batch_size=128, debug=True):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.debug = debug
        self.optimizer = None
        self.costs = []
        self._accuracies = []
        self.W = np.zeros((n_features, 1))

    def reset(self):
        self.optimizer = None
        self.costs = []
        self._accuracies = []
        self.W = np.zeros((len(self.W), 1))

    def fit(self, X, y, optimizer=LNOptimizer):
        self.costs = []
        self._accuracies = []
        BLOCK_SIZE = 0
        blocks = chunks(list(range(len(self.W))), BLOCK_SIZE)
        self.optimizer = optimizer(self.W, self.lr)
        batches = (X.shape[0] // self.batch_size)
        for i in range(self.epochs + 1):
            X, y = shuffle(X, y)
            if blocks:
                index = np.random.choice(len(blocks), replace=False)
                self.optimizer.block = blocks[index]
            for j in range(batches):
                batch = get_batch(X, y, self.batch_size, j)
                features, labels = batch
                # Foreword step
                predictions = self.forward(features)
                # Optimization step
                grad, _ = self.optimizer.optimize(labels, predictions, features)
                if blocks:
                    b = self.optimizer.block
                    self.W[b] = self.W[b] - self.lr * grad
                else:
                    self.W = self.W - self.lr * grad

            # Evaluation
            predictions = self.forward(X)
            cost = self.optimizer.loss(y, predictions)
            # acc = r2_score(y, predictions)
            self.costs.append(cost)
            # self._accuracies.append(acc)
            if i % 10 == 0 and self.debug:
                print("Epoch {}, Loss {}, Acc {}".format(i, round(cost, 4), round(acc, 4)))

    def one_epoch(self, X, y, optimizer, block):
        self.optimizer = optimizer(self.W, self.lr, block)
        if self.batch_size > 0:
            features, labels = self.get_random_batch(X, y)
        else:
            features, labels = shuffle(X, y)
        # Foreword step
        predictions = self.forward(features)
        # Optimization step
        grads, gtime = self.optimizer.optimize(labels, predictions, features)

        return grads, gtime

    def get_random_batch(self, X, y):
        sX, sy = shuffle(X, y)
        m = X.shape[0]
        if m < self.batch_size:
            self.batch_size = m
        nb_batches = (m // self.batch_size)
        j = np.random.choice(nb_batches, replace=False)
        return get_batch(sX, sy, self.batch_size, j)

    def forward(self, X):
        return X @ self.W

    def predict(self, X):
        y_pred = sigmoid(np.dot(self.W.T, X.T))
        return np.array(list(map(lambda x: 1 if x >= 0.5 else 0, y_pred.flatten())))

    def evaluate(self, X, y):
        predictions = self.forward(X)
        cost = self.optimizer.loss(y, predictions)
        acc = r2_score(y, predictions).clip(min=0)
        return cost, acc

    def summary(self, X, y):
        print("-------------------")
        print(f"Training for {self.epochs} epochs.")
        print(f">> Train:\n\tloss: {round(self.costs[-1], 4)}.")
        print(f"\taccuracy: {round(self._accuracies[-1], 3) * 100}%.")
        cost, acc = self.evaluate(X, y)
        print(f">> Test:\n\tloss: {round(cost, 4)}.")
        print(f"\taccuracy: {round(acc, 3) * 100}%.")
        return self


class SVM(object):
    """
    Support Vector Machine
    """

    def __init__(self, n_features, lr=0.01, epochs=100, batch_size=512, C=140, debug=True):
        self.lr = lr
        self.epochs = epochs
        self.C = C
        self.batch_size = batch_size
        self.optimizer = None
        self.debug = debug
        self.costs = []
        self._accuracies = []
        self.W = np.zeros((n_features, 1))

    def reset(self):
        self.optimizer = None
        self.costs = []
        self._accuracies = []
        self.W = np.zeros((len(self.W), 1))

    def fit(self, X, y, optimizer=SVMOptimizer):
        self.costs = []
        self._accuracies = []
        BLOCK_SIZE = 0
        blocks = chunks(list(range(len(self.W))), BLOCK_SIZE)
        self.optimizer = optimizer(self.W, self.lr, None, C=self.C)
        batches = (X.shape[0] // self.batch_size)
        for i in range(self.epochs + 1):
            X, y = shuffle(X, y)
            if blocks:
                index = np.random.choice(len(blocks), replace=False)
                self.optimizer.block = blocks[index]
            for j in range(batches):
                batch = get_batch(X, y, self.batch_size, j)
                features, labels = batch
                # Foreword step
                predictions = self.forward(features)
                # Optimization step
                grad, _ = self.optimizer.optimize(labels, predictions, features)
                if blocks:
                    b = self.optimizer.block
                    self.W[b] = self.W[b] - self.lr * grad
                else:
                    self.W = self.W - self.lr * grad
                self.optimizer.W = self.W
            # Evaluation
            predictions = self.forward(X)
            cost = self.optimizer.loss(y, predictions)
            acc = accuracy(y, predictions)
            self.costs.append(cost)
            self._accuracies.append(acc)
            if i % 10 == 0 and self.debug:
                print("Epoch {}, Loss {}, Acc {}".format(i, round(cost, 4), round(acc, 4)))

    def one_epoch(self, X, y, block, si):
        self.optimizer.W = self.W
        self.optimizer.block = block
        self.batch_size = si
        features, labels = self.get_random_batch(X, y)
        # Foreword step
        predictions = self.forward(features)
        # Optimization step
        grads, gtime = self.optimizer.optimize(labels, predictions, features)

        return grads, gtime

    def get_random_batch(self, X, y):
        sX, sy = shuffle(X, y)
        m = X.shape[0]
        if m < self.batch_size:
            self.batch_size = m
        nb_batches = (m // self.batch_size)
        j = np.random.choice(nb_batches, replace=False)
        return get_batch(sX, sy, self.batch_size, j)

    def forward(self, X):
        return X @ self.W

    def evaluate(self, X, y):
        predictions = self.forward(X)
        cost = self.optimizer.loss(y, predictions)
        acc = accuracy(y, predictions)
        return cost, acc

    @staticmethod
    def accuracy(y, predictions):
        y_pred = np.sign(predictions)

        return accuracy_score(y, y_pred)

    def summary(self, X, y):
        print("-------------------")
        print(f"Training for {self.epochs} epochs.")
        print(f">> Train:\n\tloss: {round(self.costs[-1], 4)}.")
        print(f"\taccuracy: {round(self._accuracies[-1], 3) * 100}%.")
        cost, acc = self.evaluate(X, y)
        print(f">> Test:\n\tloss: {round(cost, 4)}.")
        print(f"\taccuracy: {round(acc, 3) * 100}%.")
        return self


class RidgeRegression(object):

    def __init__(self, n_features, lr=0.01, epochs=1000, batch_size=128, l2_penality=32, threshold=0.5, debug=True):
        self.lr = lr
        self.epochs = epochs
        self.C = 0.5
        self.optimizer = None
        self.threshold = threshold
        self.debug = debug
        self._costs = []
        self._accuracies = []
        self.W = np.random.randn(n_features, 1) * 0.01
        self.b = 0
        self.l2_penality = l2_penality

    # Function for model training
    def fit(self, X, y):
        for epoch in range(1, self.epochs):
            # shuffle to prevent repeating update cycles
            X, Y = shuffle(X, y)
            y_pred = self.predict(X)
            self.update_weights(X, y, y_pred)

    # Helper function to update weights in gradient descent

    def update_weights(self, X, y, y_pred):
        # calculate gradients
        dW = (- (2 * X.T.dot(y - y_pred)) +
              (2 * self.l2_penality * self.W)) / X.shape[0]
        db = - 2 * np.sum(y - y_pred) / X.shape[0]

        # update weights
        self.W = self.W - self.lr * dW
        self.b = self.b - self.lr * db
        return self

    # Hypothetical function h( x )
    def predict(self, X):
        return X.dot(self.W) + self.b

    def summary(self, X, y):
        pass


if __name__ == '__main__':
    # Best Learning rate for LR is 3/4
    # Loss for lr=0.01 is: 57.8623 -- b=1
    # Loss for lr=0.001 is: 56.6784

    X_train, Y_train, X_test, Y_test = mnist(path='../datasets/mnist/', binary=False)
    lr_ = 0
    for i in range(1):
        lr_ = lr_ + 0.00001
        t = time.time()
        # m = LogisticRegression(X_train.shape[1], lr=lr_, epochs=100, batch_size=32, debug=True)
        m = MultiClassLogisticRegression(thres=1e-5, lr=lr_)
        m.fit(X_train, Y_train)
        print(m.score(X_train, Y_train))
        print(m.score(X_test, Y_test))
        # print(f"Loss for lr={m.lr} is: {round(m.costs[-1], 4)}")
        # print(f"Training done in {time.time() - t} seconds.")
        # m.summary(X_test, Y_test)

    # lr =
    # lr.fit(X_train, Y_train, lr=0.0001)
    # print(lr.weights.shape)
    # exit()
    # print(lr.score(X_train, Y_train))
