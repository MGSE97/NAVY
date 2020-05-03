import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from NeuralNetworks.Net import Net, Layer
from NeuralNetworks.Utils import Sigmoid, Linear, ReLu, Empty

Axes3D = Axes3D  # pycharm auto import

def create_graph(name):
    """
    Prepare graph window

    :param name: Window Title
    :return: ax
    """
    fig = plt.figure(name)
    fig.suptitle(name)
    ax = fig.gca()
    fig.canvas.draw_idle()
    return ax, fig

# Configuration
map = lambda a, x: a*x*(1-x)    # logistic map function
n = 1000                        # number of values
x = 1e-5 * np.ones(n)           # x values
a = np.linspace(1, 4.0, n)      # a values
iterations = 100                # iteration count

# NN
lr = 1e-3                       # learning rate
net = Net([
    Layer(n, n, Sigmoid),
])

# Draw Bifurcation diagram
g, f = create_graph("Bifurcation diagram")
for i in range(iterations):
    print("\r{}/{}".format(i, iterations), end="")
    r = map(a, x)
    g.plot(a, r, 'k', alpha=max(1/iterations, 0.01))
    #g.plot(a, r, 'k')

    # teach NN
    nr = net.forward(x)
    net.backwards(r, nr, lr)

    # update x
    x = r

plt.pause(0.1)

# Draw NN Bifurcation diagram
g2, f2 = create_graph("NN Bifurcation diagram")
x = 1e-5 * np.ones(n)
for i in range(iterations):
    print("\r{}/{}".format(i, iterations), end="")
    x = net.forward(x)
    #g2.plot(a, x, 'k', alpha=1/iterations)
    g2.plot(a, x, 'k')


plt.show()
