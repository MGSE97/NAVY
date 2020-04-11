import numpy as np
from mpl_toolkits.mplot3d import Axes3D

Axes3D = Axes3D  # pycharm auto import
import matplotlib.pyplot as plt


def transform(position, transformation):
    """
    Transform position using transformation

    :param position: X,Y,Z position
    :param transformation: Model Transformation
    :return: New X,Y,Z position
    """
    p, a, b, c, d, e, f, g, h, i, j, k, l = transformation
    return np.array([a, b, c, d, e, f, g, h, i]).reshape(3, 3) @ np.array(position).reshape(3, 1) + np.array(
        [j, k, l]).reshape(3, 1)


def pick_transform(transforms):
    """
    Pick random transformation usign probabilities

    :param transforms: Model Transformations
    :return: Selected Transformation
    """
    probabilities = [t[0] for t in transforms]
    return transforms[int(np.random.choice(range(0, len(transforms)), 1, p=probabilities))]


def create_graph(name):
    """
    Prepare graph window

    :param name: Window Title
    :return: ax
    """
    fig = plt.figure(name)
    fig.suptitle(name)
    ax = fig.gca(projection='3d')
    fig.canvas.draw_idle()
    return ax


def draw_points(ax, positions):
    """
    Draw points to graph.

    :param ax: graph
    :param positions: position in format [[x,][y,][z,]]
    """
    x, y, z = positions
    ax.scatter(x, y, z, c="green", s=3)


def draw_model(name, ax, model, points_count):
    """
    Draw model to graph

    :param ax: graph
    :param model: IFS model
    :param points_count: number of points to draw
    """
    position = [0, 0, 0]
    positions = [[position[0]], [position[1]], [position[2]]]

    pd = int(points_count / 100)  # print % points step
    for x in range(0, points_count):
        position = transform(position, pick_transform(model))
        # save generated point
        positions[0].append(position.item(0))
        positions[1].append(position.item(1))
        positions[2].append(position.item(2))
        if x % pd == 0:
            print("\r{:3d}% ... {}".format(int(x / pd), name), end="")

    print("\r100% ... {}".format(name))

    # draw points
    draw_points(ax, positions)
    plt.pause(0.001)


# Configuration
points = 100000

# probability,  a      b     c     d     e     f     g      h    i     j     k     l
model1 = [
    [0.40, 0.00, 0.00, 0.01, 0.00, 0.26, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00, 0.00],
    [0.15, 0.20, -0.26, -0.01, 0.23, 0.22, -0.07, 0.07, 0.00, 0.24, 0.00, 0.80, 0.00],
    [0.15, -0.25, 0.28, 0.01, 0.26, 0.24, -0.07, 0.07, 0.00, 0.24, 0.00, 0.22, 0.00],
    [0.30, 0.85, 0.04, -0.01, -0.04, 0.85, 0.09, 0.00, 0.08, 0.84, 0.00, 0.80, 0.00]
]
model2 = [
    [0.25, 0.05, 0.00, 0.00, 0.00, 0.60, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00, 0.00],
    [0.25, 0.45, -0.22, 0.22, 0.22, 0.45, 0.22, -0.22, 0.22, -0.45, 0.00, 1.00, 0.00],
    [0.25, -0.45, 0.22, -0.22, 0.22, 0.45, 0.22, 0.22, -0.22, 0.45, 0.00, 1.25, 0.00],
    [0.25, 0.49, -0.08, 0.08, 0.08, 0.49, 0.08, 0.08, -0.08, 0.49, 0.00, 2.00, 0.00]
]

# Model 1 drawing
g1 = create_graph("Model 1")
draw_model("Model 1", g1, model1, points)

# Model 2 drawing
g2 = create_graph("Model 2")
draw_model("Model 2", g2, model2, int(points / 10))
plt.show()
