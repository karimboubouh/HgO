import time

import numpy as np

from src.utils import sigmoid_prime, number_coordinates, elog, log


class DNNOptimizer(object):
    """
    Deep Neural Network optimizer using coordinate descent (CD) and gradient descent (GD)

    Arguments:
    W -- weights, -------------
    lr -- learning rate of the gradient descent update rule
    block -- block of coordinates to update if
    """

    def __init__(self, W, b, lr=0.01, block=None):
        self.W = W
        self.b = b
        self.lr = lr
        self.block = block
        self.grads = []

    def optimize(self, y, A, Zs):
        subA = [a[self.block[i], :] for i, a in enumerate(A)]
        subAZs = [z[self.block[i + 1], :] for i, z in enumerate(Zs)]
        subW = [w[self.block[i + 1], :][:, self.block[i]] for i, w in enumerate(self.W)]
        subB = [b[self.block[i + 1], :] for i, b in enumerate(self.b)]
        dw, db, gtime = self.coordinate_descent(y, subA, subAZs, subW, subB)

        return dw, db, gtime

    def gradient_descent(self, y, A, Zs):
        m = y.shape[1]
        t = time.time()
        num_layers = len(self.b) + 1
        db = [np.array([])] * len(self.b)
        dw = [np.array([])] * len(self.W)
        error = self.cost_derivative(A[-1], y) * sigmoid_prime(Zs[-1])
        db[-1] = 1. / m * np.sum(error, axis=1, keepdims=True)
        dw[-1] = 1. / m * np.dot(error, A[-2].transpose())
        for layer in range(2, num_layers):
            error = np.dot(self.W[-layer + 1].transpose(), error) * sigmoid_prime(Zs[-layer])
            db[-layer] = 1. / m * np.sum(error, axis=1, keepdims=True)
            dw[-layer] = 1. / m * np.dot(error, A[-layer - 1].transpose())
        gtime = time.time() - t

        return dw, db, gtime

    def coordinate_descent(self, y, A, Zs, W, b):
        m = y.shape[1]
        t = time.time()
        num_layers = len(b) + 1
        db = [np.array([])] * len(b)
        dw = [np.array([])] * len(W)
        error = self.cost_derivative(A[-1], y) * sigmoid_prime(Zs[-1])
        db[-1] = 1. / m * np.sum(error, axis=1, keepdims=True)
        dw[-1] = 1. / m * np.dot(error, A[-2].transpose())
        for layer in range(2, num_layers):
            error = np.dot(W[-layer + 1].transpose(), error) * sigmoid_prime(Zs[-layer])
            db[-layer] = 1. / m * np.sum(error, axis=1, keepdims=True)
            dw[-layer] = 1. / m * np.dot(error, A[-layer - 1].transpose())
        gtime = time.time() - t

        return dw, db, gtime

    @staticmethod
    def cost_derivative(output_activations, y):
        """Return the vector of partial derivatives for the output activations."""
        return output_activations - y
