import copy
import time

import numpy as np

from src.utils import sigmoid


class LROptimizer(object):
    """
    Logistic regression optimizer using coordinate descent (CD) and gradient descent (GD)

    Arguments:
    W -- weights, a numpy array of size (n_x, 1)
    block -- block of coordinates to update if
    """

    def __init__(self, W, block=None):
        self.W = W
        self.block = block
        self.grads = []

    def optimize(self, y, y_pred, X):
        if self.block is None:
            grad, gtime = self.gradient_descent(y, y_pred, X)
        else:
            subX = X[:, self.block]
            grad, gtime = self.coordinate_descent(y, y_pred, subX)
        return grad, gtime

    def estimate_lr_GD(self, X, y):
        maxi = []
        for i in range(100):
            w1 = np.random.rand(X.shape[1], 1)
            y_pred = sigmoid(X @ w1)
            dw1, gtime = self.gradient_descent(y, y_pred, X)
            w2 = np.random.rand(X.shape[1], 1)
            y_pred = sigmoid(X @ w2)
            dw2, gtime = self.gradient_descent(y, y_pred, X)
            L = np.linalg.norm(dw1 - dw2) / np.linalg.norm(w1 - w2)
            maxi.append(L)
        print(f"MAX == {np.max(maxi)}")

    def estimate_lr_CD(self, X, y, block):
        self.block = block
        maxi = []
        X = X[:, self.block]
        for i in range(100):
            w1 = np.random.rand(X.shape[1], 1)
            y_pred = sigmoid(X @ w1)
            dw1, gtime = self.coordinate_descent(y, y_pred, X)
            w2 = np.random.rand(X.shape[1], 1)
            y_pred = sigmoid(X @ w2)
            dw2, gtime = self.coordinate_descent(y, y_pred, X)
            L = np.linalg.norm(dw1 - dw2) / np.linalg.norm(w1 - w2)
            maxi.append(L)
        print(f"MAX == {np.max(maxi)}")

    @staticmethod
    def loss(y, y_pred):
        return -(1.0 / len(y)) * np.sum(y * np.log(y_pred) + (1.0 - y) * np.log(1.0 - y_pred + 1e-7))

    @staticmethod
    def gradient_descent(y, y_pred, X):
        t = time.time()
        m = X.shape[0]
        dw = 1 / m * X.T @ (y_pred - y)
        gtime = time.time() - t
        return dw, gtime

    @staticmethod
    def coordinate_descent(y, y_pred, X):
        t = time.time()
        m = X.shape[0]
        dw = 1 / m * X.T @ (y_pred - y)
        gtime = time.time() - t
        return dw, gtime


class LNOptimizer(object):
    """
    Linear regression optimizer using coordinate descent (CD) and gradient descent (GD)

    Arguments:
    W -- weights, a numpy array of size (n_x, 1)
    lr -- learning rate of the gradient descent update rule
    block -- block of coordinates to update if
    """

    def __init__(self, W, lr=0.01, block=None):
        self.W = W
        self.lr = lr
        self.block = block

    def optimize(self, y, y_pred, X):
        if self.block is None:
            grad, gtime = self.gradient_descent(y, y_pred, X)
        else:
            subX = copy.deepcopy(X[:, self.block])
            grad, gtime = self.coordinate_descent(y, y_pred, subX)

        return grad, gtime

    @staticmethod
    def loss(y, y_pred):
        return np.sum((y_pred - y) ** 2) / (2 * len(y))

    @staticmethod
    def gradient_descent(y, y_pred, X):
        t = time.time()
        m = X.shape[0]
        dw = 1 / m * X.T @ (y_pred - y)
        gtime = time.time() - t
        return dw, gtime

    @staticmethod
    def coordinate_descent(y, y_pred, X):
        t = time.time()
        m = X.shape[0]
        dw = 1 / m * X.T @ (y_pred - y)
        gtime = time.time() - t
        return dw, gtime


class RROptimizer(object):
    """
    Ridge Regression optimizer using coordinate descent (CD) and gradient descent (GD)

    Arguments:
    W -- weights, a numpy array of size (n_x, 1)
    lr -- learning rate of the gradient descent update rule
    block -- block of coordinates to update if
    """

    def __init__(self, W, lr=0.001, block=None):
        self.W = W
        self.lr = lr
        self.block = block

    def optimize(self, y, y_pred, X):
        if self.block is None:
            grad = self.gradient_descent(y, y_pred, X)
            self.W = self.W - self.lr * grad
        else:
            grad = self.coordinate_descent(y, y_pred, X[:, self.block])
            self.W[self.block] = self.W[self.block] - self.lr * grad

        return self.W, grad

    @staticmethod
    def loss(y, y_pred):
        return -(1 / len(y)) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    @staticmethod
    def gradient_descent(y, y_pred, X):
        return np.dot((y_pred - y), X)[0].reshape(-1, 1)

    @staticmethod
    def coordinate_descent(y, y_pred, X):
        return np.dot((y_pred - y), X)[0].reshape(-1, 1)


class SVMOptimizer(object):
    """
    Support Vector Machine (SVM) optimizer using coordinate descent (CD) and gradient descent (GD)

    Arguments:
    W -- weights, a numpy array of size (n_x, 1)
    lr -- learning rate of the gradient descent update rule
    block -- block of coordinates to update if
    """

    def __init__(self, W, lr=0.00001, block=None, C=1000):
        self.W = W
        self.lr = lr
        self.block = block
        self.C = C

    def loss(self, y, y_pred):
        # calculate hinge loss
        N = len(y_pred)
        distances = 1 - y * y_pred
        distances[distances < 0] = 0  # equivalent to max(0, distance)
        hinge_loss = self.C * (np.sum(distances) / N)
        # calculate cost
        W2 = self.W.ravel() @ self.W.ravel()
        cost = 1 / 2 * W2 + hinge_loss
        return cost

    def optimize(self, y, y_pred, X):
        if self.block is None:
            grad, gtime = self.gradient_descent(y, y_pred, X)
        else:
            subX = X[:, self.block]
            W = self.W[self.block]
            grad, gtime = self.coordinate_descent(y, y_pred, subX, W)

        return grad, gtime

    def gradient_descent(self, y, y_pred, X):
        t = time.time()
        distance = 1 - (y * y_pred)
        dw = np.zeros((len(self.W), 1))
        for ind, d in enumerate(distance):
            if max(0, d) == 0:
                di = self.W
            else:
                di = self.W - (self.C * y[ind] * X[ind]).reshape(-1, 1)
            dw += di
        dw = dw / len(X)
        gtime = time.time() - t
        return dw, gtime

    def coordinate_descent(self, y, y_pred, X, W):
        t = time.time()
        distance = 1 - (y * y_pred)
        dw = np.zeros((len(W), 1))
        for ind, d in enumerate(distance):
            if max(0, d) == 0:
                di = W
            else:
                di = W - (self.C * y[ind] * X[ind]).reshape(-1, 1)
            dw += di
        dw = dw / len(X)
        gtime = time.time() - t
        return dw, gtime
