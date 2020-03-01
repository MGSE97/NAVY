import numpy as np

from Perceptron import Perceptron


class Net:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        data = inputs
        for layer in self.layers:
            data = layer.forward(data)

        return data

    def backwards(self, error, learning_rate):
        for layer in self.layers:
            layer.backwards(error, learning_rate)


class Layer:
    def __init__(self, inputs_size, outputs_size, activation_func):
        self.inputs_size = inputs_size
        self.outputs_size = outputs_size
        self.neurons = [Perceptron(activation_func, inputs_size) for i in range(0, outputs_size)]
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return [neuron.forward(inputs) for neuron in self.neurons]

    def backwards(self, error, learning_rate):
        for neuron in self.neurons:
            neuron.backwards(error, learning_rate, self.inputs)


def grad_error(real, guess):
    return np.abs(0.5 * (real - guess) ** 2)
