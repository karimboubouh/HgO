import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

from src.datasets import get_dataset, mnist
from src.nonlinear_optimizers import DNNOptimizer
from src.utils import sigmoid, nn_chunks, get_block, Map


class DNN(object):
    """Initialize a Deep Neural Network (DNN) model

    Parameters
    ---------
    sizes: list, optional
        A list of numbers specifying number of neurons in each layer. Not
        required if pretrained model is used.

    learning_rate: float, optional
        learning rate for the gradient descent optimization. Defaults to 3.0

    batch_size: int, optional
        Size of the mini batch of training examples as used by Stochastic
        Gradient Descent. Denotes after how many examples the weights and biases
        would be updated. Default size is 10.

    epochs: int, optional
        Number of Epochs through which Neural Network will be trained. Defaults
        to 10.

    cost: string, optional
        The cost function to be used to evaluate by how much our assumptions
        were deviated. Defaults to `rms`.

    lmbda: float, optional
        The L1 regularization parameter to generalize the neural network.
        Default value is 0.0.

    """

    def __init__(self, layer_dims, epochs=20, lr=3.0, batch_size=64):
        self.num_layers = len(layer_dims)
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.optimizer = None
        self.b = [np.random.randn(y, 1) for y in layer_dims[1:]]
        self.W = [np.random.randn(y, x) for y, x in zip(layer_dims[1:], layer_dims[:-1])]
        self._costs = []
        self._accuracies = []

    def reset(self):
        self.b = [np.random.randn(y.shape[0], 1) for y in self.b]
        self.W = [np.random.randn(x.shape[0], x.shape[1]) for x in self.W]
        self._costs = []
        self._accuracies = []

    def fit(self, _X, _y, bz, optimizer=DNNOptimizer):
        acc = []
        avg_time = []
        # BLOCK_SIZES = [28, 14, 10]
        # BLOCK_SIZES = [784, 30, 10]
        # BLOCK_SIZES = []
        blocks = nn_chunks(self.W, bz)
        self.optimizer = optimizer(self.W, self.b, self.lr)
        batches = (_X.shape[0] // self.batch_size)

        for i in range(self.epochs):
            epoch_time = 0
            X, y = shuffle(_X, _y)
            X, y = X.T, y.T
            for j in range(batches):
                bl = get_block(blocks)
                self.optimizer.block = bl
                batch = self.get_batch(X, y, j)
                features, labels = batch
                # Foreword step
                A, Zs = self.forward(features)
                # Optimization step
                dw, db, gtime = self.optimizer.optimize(labels, A, Zs)
                epoch_time += gtime
                if blocks:
                    for idx, (w, b, gw, gb) in enumerate(zip(self.W, self.b, dw, db)):
                        w[np.ix_(bl[idx + 1], bl[idx])] -= self.lr * gw
                        b[np.ix_(bl[idx + 1])] -= self.lr * gb
                else:
                    self.W = [w - self.lr * gw for w, gw in zip(self.W, dw)]
                    self.b = [b - self.lr * gb for b, gb in zip(self.b, db)]
            # Evaluation
            out, _ = self.evaluate(X.T, y.T, convert=True)
            acc.append(out)
            avg_time.append(epoch_time)
            print(f"Epoch {i + 1} done in {round(epoch_time, 4)}s with accuracy {round(out, 4) * 100} %.")
        print()
        print(f"AVG time: {np.mean(avg_time)}")
        print()
        return acc

    def one_epoch(self, X, y, block, si):
        self.optimizer.W = self.W
        self.optimizer.b = self.b
        self.optimizer.block = block
        self.batch_size = si
        features, labels = self.get_random_batch(X, y)
        # Foreword step
        A, Zs = self.forward(features)
        # Optimization step
        dw, db, gtime = self.optimizer.optimize(labels, A, Zs)
        return (dw, db), gtime

    def get_batch(self, X, y, j):
        begin = j * self.batch_size
        end = min(begin + self.batch_size, X.shape[1])
        if end + self.batch_size > X.shape[1]:
            end = X.shape[1]
        X_ = X[:, begin:end]
        y_ = y[:, begin:end]
        return X_, y_

    def get_random_batch(self, X, y):
        sX, sy = shuffle(X, y)
        nb_batches = (sX.shape[0] // self.batch_size)
        if nb_batches > 0:
            j = np.random.choice(nb_batches, replace=False)
            _X, _y = self.get_batch(sX.T, sy.T, j)
        else:
            _X, _y = sX, sy
        return _X, _y

    def forward(self, X):
        A = [X]  # list to store activations, layer by layer
        Zs = []  # list to store Z vectors, layer by layer
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            Z = w @ A[i] + b
            assert (Z.shape == (w.shape[0], A[i].shape[1]))
            Zs.append(Z)
            A.append(sigmoid(Z))

        return A, Zs

    def evaluate(self, X, y, convert=True):
        Xt = X.T
        activations, _ = self.forward(Xt)
        y_pred = np.argmax(activations[-1], axis=0)
        if convert:
            acc = np.equal(np.argmax(y, axis=-1), y_pred).mean()
        else:
            acc = np.equal(y, y_pred).mean()

        return acc, acc

    def summary(self, X, y):
        pass


if __name__ == '__main__':
    np.random.seed(10)

    X_train, Y_train, X_test, Y_test = mnist(path='../datasets/mnist/', binary=False)
    lr_ = 25
    for i in range(20):
        lr_ = lr_ + 1
        dnn = DNN([784, 30, 10], epochs=5, batch_size=128, lr=lr_)
        acc = dnn.fit(X_train, Y_train, [128, 10, 10])
        print(f"for lr={lr_}, acc ={dnn.evaluate(X_test, Y_test)[0]}")
