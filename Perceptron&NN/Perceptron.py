import numpy as np


class Perceptron:
    def __init__(self, activation_func, size):
        self.activation_func = activation_func
        self.size = size
        self.weights = np.random.uniform(-1, 1, size)
        self.bias = np.random.uniform(-1, 1, 1)[0]

    def forward(self, inputs):
        # bias
        y = self.bias
        # sum(xi * wi)
        for i in range(0, self.size):
            y += inputs[i] * self.weights[i]
        # sum(xi*wi) + bias
        return self.activation_func(y)

    def backwards(self, error, learning_rate, inputs):
        # recalculate weights
        for i in range(0, self.size):
            self.weights[i] = Perceptron.weight_recalc(
                self.weights[i],
                error,
                inputs[i],
                learning_rate
            )
        # recalculate bias
        self.bias = self.bias + error*learning_rate

    @staticmethod
    def weight_recalc(weight, error, input, lerning_rate):
        return weight + error * input * lerning_rate


def perceptron_simple_error(real, guess):
    return np.abs(real - guess)
