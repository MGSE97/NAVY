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

    def backwards(self, reals, guess, learning_rate):
        error = np.subtract(guess, reals)
        for layer in reversed(self.layers):
            error = layer.backwards(error, learning_rate)

    def __str__(self):
        return str.join('\n', [l.__str__() for l in self.layers])

class Layer:
    def __init__(self, inputs_size, outputs_size, activation_func):
        self.inputs_size = inputs_size
        self.outputs_size = outputs_size
        self.activation_func = activation_func
        # Wjk * Aj = O
        self.weights = np.random.uniform(0, 1, size=(outputs_size, inputs_size))
        self.biases = np.random.uniform(0, 1, outputs_size) #np.ones(outputs_size)
        self.inputs = None
        self.nets = None
        self.outputs = None

    def forward(self, inputs):
        self.inputs = inputs
        #inputs_t = np.transpose(self.inputs)
        self.nets = [
            np.sum(
                self.weights[y:] @ inputs
            ) +
            self.biases[y]
            for y in range(0, self.outputs_size)
        ]
        self.outputs = [self.activation_func.forward(n) for n in self.nets]
        return self.outputs

    def backwards(self, error, learning_rate):
        tmp = error * [self.activation_func.backwards(o) for o in self.outputs]
        #print("tmp ", tmp.shape)
        #print("error ", error.shape)
        #print("weights ", self.weights.shape)
        error = self.weights.transpose() @ tmp
        tmp = learning_rate * tmp
        #print("tmp ", tmp.reshape(tmp.shape[0], 1).shape)
        #print("inputs ", np.array(self.inputs).reshape(1,self.inputs_size).shape)
        self.weights -= tmp.reshape(tmp.shape[0], 1) * np.array(self.inputs).reshape(1, self.inputs_size)
        self.biases -= tmp
        return error

    def __str__(self):
        return '{}:{}>\n{}\n{}'.format(self.inputs_size, self.outputs_size, self.biases, self.weights)

def grad_error(real, guess):
    return np.abs(0.5 * (real - guess) ** 2)

