import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

Axes3D = Axes3D  # pycharm auto import

from Perceptron import Perceptron, perceptron_simple_error
from Utils import rand_point_arr, Point, prepareView

def drawView(ax, data_x, data_y, color):
    ax.plot(data_x, data_y, 'ro', color=color, antialiased=True, alpha=0.5)
    plt.pause(0.0001)

def drawPoints(ax, data):
    drawView(ax, [p.data[0] for p in data if p.label == 0], [p.data[1] for p in data if p.label == 0], 'black')
    drawView(ax, [p.data[0] for p in data if p.label > 0], [p.data[1] for p in data if p.label > 0], 'red')
    drawView(ax, [p.data[0] for p in data if p.label < 0], [p.data[1] for p in data if p.label < 0], 'blue')

def drawFunction(ax, line):
    mx = 25
    ax.plot(range(-mx, mx), [line(x) for x in range(-mx, mx)])

def drawLine(ax, a, b):
    ax.plot([a[0], b[0]], [a[1], b[1]])

def normalize(point):
    l = np.sqrt(sum([p**2 for p in point]))
    result = Point()
    for p in point:
        result.data.append(p/l)
    return result

def test_func(point, func):
    vN = [-1, func(-1)]
    vP = [1, func(1)]
    return np.sign(np.cross(point - vN, np.subtract(vP, vN)))

def gen_labeled(input_size):
    t = rand_point_arr(input_size)
    for p in t:
        p.label = test_func(p.data, line)
    return t

# prepare
net = Perceptron(np.sign, 2)

input_size = 1000
epochs = 10
learning_rate = 0.1

line = lambda x: 4 * x + 5
teach = gen_labeled(input_size)
test = gen_labeled(input_size)

# view
viewTeach = prepareView("Perceptron Teach")
drawFunction(viewTeach, line)
drawPoints(viewTeach, teach)

viewTest = prepareView("Perceptron Test")
drawFunction(viewTest, line)

# learn
for epoch in range(0, epochs):
    #teach = gen_labeled(input_size)
    #drawPoints(viewTeach, teach)
    losses = 0
    for point in teach:
        y = net.forward(point.data)
        loss = perceptron_simple_error(point.label, y)
        net.backwards(loss, learning_rate, point.data)
        losses += loss
    print("Loss({}): {}".format(epoch, losses / len(teach)))
    #if losses == 0.0:
        #break

# test
losses = 0
for point in test:
    y = net.forward(point.data)
    loss = perceptron_simple_error(point.label, y)
    point.label = y
    losses += loss
print("Loss(test): {}".format(losses / len(teach)))

drawPoints(viewTest, test)

plt.show()
