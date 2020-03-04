import copy

import numpy as np
from Net import Layer, Net, grad_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from Perceptron import perceptron_simple_error
from Utils import rand_point_arr, prepareView, Sigmoid

Axes3D = Axes3D  # pycharm auto import


def drawView(ax, data_x, data_y, color):
    return ax.plot(data_x, data_y, 'ro', color=color, antialiased=True, alpha=0.5)[0]


def drawPoints(ax, data, points = None):
    arr = []
    if points != None:
        for p in points:
            p.remove()
        points.clear()
    arr.append(drawView(ax, [p.data[0] for p in data if p.label <= 0.5], [p.data[1] for p in data if p.label <= 0.5], 'red'))
    arr.append(drawView(ax, [p.data[0] for p in data if p.label > 0.5], [p.data[1] for p in data if p.label > 0.5], 'green'))
    plt.pause(0.0001)
    return arr


def gen_labeled(input_size):
    t = rand_point_arr(input_size, 0, 1)
    for p in t:
        p.label = test_func(p.data)
        p.originalLabel = p.label
    return t


def test_func(data):
    x, y = [round(x) for x in data]
    return x != y


# prepare
net = Net([
    Layer(2, 2, Sigmoid),
    Layer(2, 1, Sigmoid)
])

input_size = 10000
epochs = 1000
draw_epoch = 5
learning_rate = 0.1

teach = gen_labeled(input_size)
test = gen_labeled((int)(input_size/10))


# view
#viewTeach = prepareView("XOR Teach")
#drawPoints(viewTeach, teach)
viewTest = prepareView("XOR Test")
points = drawPoints(viewTest, test)

# learn
for epoch in range(0, epochs):
    losses = 0
    for point in teach:
        y = net.forward(point.data)
        #loss = perceptron_simple_error(point.label, y[0])
        net.backwards([point.label], y, learning_rate)
        #losses += loss
    #print("Loss({}): {}".format(epoch, losses / len(teach)))
    #print(net)

    if epoch % draw_epoch == 0:
        # test
        #losses_t = 0
        for point in test:
            y = net.forward(point.data)[0]
            #loss = perceptron_simple_error(point.label, y)
            point.label = y
            #losses_t += loss
        #print("Loss({}): {}".format(epoch, losses_t / len(test)))
        print(epoch)
        points = drawPoints(viewTest, test, points)

plt.show()
