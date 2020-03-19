from Net import Layer, Net
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
net_size = 8
net = Net([
    Layer(2, net_size, Sigmoid),
    Layer(net_size, 1, Sigmoid)
])

input_size = 5000
test_size = 2000
#epochs = 1000
draw_epoch = 1
learning_rate = 0.05
# ns: 16 fail       is: 1000 de: 5 lr: 0.1
# ns: 8  e: 105     is: 1000 de: 5 lr: 1.0 unstable
# ns: 8  e: 150     is: 1000 de: 5 lr: 0.5
# ns: 8  e: 135-45  is: 1000 de: 5 lr: 0.2
# ns: 8  e: 445     is: 1000 de: 5 lr: 0.1
# ns: 8  fail       is: 1000 de: 5 lr: 0.01
# ns: 6  fail       is: 1000 de: 5 lr: 0.5
# ns: 6  fail       is: 1000 de: 5 lr: 0.1
# ns: 5  fail       is: 1000 de: 5 lr: 0.5
# ns: 5  fail       is: 1000 de: 5 lr: 0.1
# ns: 4  fail       is: 1000 de: 5 lr: 0.5
# ns: 4  fail       is: 1000 de: 5 lr: 0.1
# ns: 2  fail       is: 1000 de: 5 lr: 0.5
# ns: 2  fail       is: 1000 de: 5 lr: 0.1

# ns: 16 fail       is: 10000 de: 5 lr: 0.1
# ns: 8  e: 25      is: 10000 de: 5 lr: 0.1
# ns: 8  e: 112     is: 10000 de: 2 lr: 0.01

teach = gen_labeled(input_size)
test = gen_labeled(test_size)


# view
#viewTeach = prepareView("XOR Teach")
#drawPoints(viewTeach, teach)
viewTest = prepareView("XOR Test")
points = drawPoints(viewTest, test)

# learn
#for epoch in range(0, epochs):
epoch = 0
while(True):
    for point in teach:
        y = net.forward(point.data)
        net.backwards([point.label], y, learning_rate)

    if epoch % draw_epoch == 0:
        # test
        losses = 0
        for point in test:
            y = net.forward(point.data)[0]
            loss = 0.5*(point.label - y)**2
            point.label = y
            losses += loss
        print("Loss({}): {}".format(epoch, losses))
        #print(epoch)
        points = drawPoints(viewTest, test, points)
        #net.backwards([0], 1, 10.0)
        
    epoch += 1

plt.show()
