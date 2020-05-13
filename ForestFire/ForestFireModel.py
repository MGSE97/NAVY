import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

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


def draw_forest(ax, fig, forest, colors):
    """
    Draw forest to graph.

    :param ax: graph
    :param forest: forest in format [[c,...],...]
    :param colors: color values of states c
    """
    x, y, z = [], [], []
    for ix, row in enumerate(forest):
        for iy, col in enumerate(row):
            x.append(ix)
            y.append(iy)
            z.append(colors[col])

    ax.cla()
    ax.scatter(x, y, c=z, s=5)
    fig.canvas.draw()
    fig.canvas.flush_events()


def init_forest(size, p, f):
    return np.random.choice((0, 1, 2), size=size, p=(p, 1.0-p-f, f))


def update_forest(forest, size, p, f):
    new_forest = forest.copy()

    for x, row in enumerate(forest):
        for y, col in enumerate(row):
            val = col

            if val == 0 or val == 3:
                val = np.random.choice((0, 1), p=(1.0-p, p))
            elif val == 1:
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if size[0] > x + dx >= 0 and size[1] > y + dy >= 0 and forest[x + dx, y + dy] == 2:
                            val = 2
                            break
                    if val == 2:
                        break
                if val == 1:
                    val = np.random.choice((1, 2), p=(1.0-f, f))
            elif val == 2:
                val = 3

            new_forest[x, y] = val

    return new_forest


if __name__ == '__main__':
    # Configure
    density = 0.5
    size = (100, 100)
    p, f = 0.05, 0.001

    # Crate forest
    states = [
        (0.0, 0.0, 0.0), # Empty
        (0.0, 1.0, 0.0), # Alive
        (1.0, 0.0, 0.0), # Burning
        (1.0, 1.0, 1.0), # Burned
    ]
    forest = init_forest(size, density, f)

    # draw
    g, fig = create_graph("Forest Fire Model")
    draw_forest(g, fig, forest, states)
    plt.pause(1)

    # Update
    while True:
        forest = update_forest(forest, size, p, f)
        draw_forest(g, fig, forest, states)

    # Show
    plt.show()