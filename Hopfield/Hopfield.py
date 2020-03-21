import numpy as np


class HopfieldNet:
    def __init__(self, size_x, size_y):
        self.size = (size_x*size_y, size_x*size_y)
        self.weights = np.zeros(shape=self.size)
        self.ones = np.zeros(shape=self.size)
        np.fill_diagonal(self.ones, 1)
        self.empty = True

    def to_vec(self, pattern):
        """
        Flattens N*M to array

        :param pattern: Pattern to convert
        :return: Flattened Pattern
        """
        return pattern.reshape(1, pattern.data.shape[0]*pattern.data.shape[1])

    def from_vec(self, data, size):
        """
        Converts flattened data to M*N matrix

        :param data: Array
        :param size: Size to convert
        :return: M*N Matrix
        """
        return data.reshape(size[0], size[1])

    def save(self, pattern):
        """
        Saves pattern to network

        :param pattern: Pattern to save
        """
        p = self.to_vec(pattern)
        w = (p * p.transpose()) - self.ones
        if self.empty:
            self.weights = w
            self.empty = False
        else:
            self.weights += w

    def recover(self, pattern, sync=True):
        """
        Recovers pattern using network

        :param pattern: Destroyed pattern
        :param sync: Method Synchronous/Asynchronous
        :return: Recovered pattern
        """
        p = self.to_vec(pattern)
        if sync:
            p = self.weights @ p.transpose()
        else:
            for i, col in enumerate(self.weights):
                x = np.sign(p @ self.weights[:, i]).astype(int)
                p[0][i] = x[0]

        return self.from_vec(np.sign(p).astype(int), pattern.shape)

