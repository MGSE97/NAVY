import numpy as np
from Net import Layer, Net, grad_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from Utils import rand_point_arr, prepareView

Axes3D = Axes3D  # pycharm auto import


def drawView(ax, data_x, data_y, color):
    ax.plot(data_x, data_y, 'ro', color=color, antialiased=True, alpha=0.5)
    plt.pause(0.0001)


def drawPoints(ax, data):
    drawView(ax, [p.data[0] for p in data if p.label == 0], [p.data[1] for p in data if p.label == 0], 'red')
    drawView(ax, [p.data[0] for p in data if p.label == 1], [p.data[1] for p in data if p.label == 1], 'green')


def gen_labeled(input_size):
    t = rand_point_arr(input_size, 0, 1)
    for p in t:
        p.label = test_func(p.data)
    return t


def test_func(data):
    x, y = [round(x) for x in data]
    return x != y


# prepare
net = Net([
    Layer(2, 2, np.sign),
    Layer(2, 1, np.sign)
])

input_size = 1000
epochs = 10
learning_rate = 0.1

teach = gen_labeled(input_size)
test = gen_labeled(input_size)


# view
viewTeach = prepareView("XOR Teach")
drawPoints(viewTeach, teach)

# learn
for epoch in range(0, epochs):
    losses = 0
    for point in teach:
        y = net.forward(point.data)[0]
        loss = grad_error(point.label, y)
        net.backwards(loss, learning_rate)
        losses += loss
    print("Loss({}): {}".format(epoch, losses / len(teach)))

# test
losses = 0
for point in test:
    y = net.forward(point.data)[0]
    loss = grad_error(point.label, y)
    point.label = y
    losses += loss
print("Loss(test): {}".format(losses / len(teach)))


viewTest = prepareView("XOR Test")
drawPoints(viewTest, test)


plt.show()
