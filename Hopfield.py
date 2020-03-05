import numpy as np


class HopfieldNet:
    def __init__(self, size_x, size_y):
        self.size = (size_x*size_y, size_x*size_y)
        self.weights = np.zeros(shape=self.size)
        self.ones = np.zeros(shape=self.size)
        np.fill_diagonal(self.ones, 1)
        self.empty = True

    def to_vec(self, pattern):
        return pattern.reshape(1, pattern.data.shape[0]*pattern.data.shape[1])

    def from_vec(self, data, size):
        return data.reshape(size[0], size[1])

    def save(self, pattern):
        p = self.to_vec(pattern)
        w = (p * p.transpose()) - self.ones
        if self.empty:
            self.weights = w
            self.empty = False
        else:
            self.weights += w

    def recover(self, pattern):
        #sync
        p = self.to_vec(pattern)
        return self.from_vec(np.sign(self.weights @ p.transpose()).astype(int), pattern.shape)
