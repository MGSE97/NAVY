import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
from matplotlib.patches import Circle

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


def draw_step(ax, line, sizes):
    """
    Draw pendulum step to graph.

    :param sizes: maximal lengths of line
    :param ax: graph
    :param line: line in format [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = line

    # Clear & Resize
    ax.cla()
    size = np.sum(sizes) + 1
    g.axis([-size, size, -size, size])
    g.autoscale(False)

    # Plot step
    ax.plot([0, x1, x2], [0, y1, y2], lw=2, c='k')
    ax.add_patch(Circle((0, 0), 0.05, fc='k', zorder=10))
    ax.add_patch(Circle((x1, y1), 0.08, fc='b', ec='b', zorder=10))
    ax.add_patch(Circle((x2, y2), 0.08, fc='r', ec='r', zorder=10))


def get_derivative(y, t, l1, l2, m1, m2):
    # Derivatives of first y
    o1, v1, o2, v2 = y
    # Acceleration calculation for 1st and 2nd pendulum
    o1d = v1
    v1d = (m2 * g * np.sin(o2) * np.cos(o1 - o2) - m2 * np.sin(o1 - o2) * (
                l1 * v1 ** 2 * np.cos(o1 - o2) + l2 * v2 ** 2) - (m1 + m2) * g * np.sin(o1)) \
          / (l1 * (m1 + m2 * np.sin(o1 - o2) ** 2))
    o2d = v2
    v2d = ((m1 + m2) * (l1 * v1 ** 2 * np.sin(o1 - o2) - g * np.sin(o2) + g * np.sin(o1) * np.cos(
                o1 - o2)) + m2 * l2 * v2 ** 2 * np.sin(o1 - o2) * np.cos(o1 - o2)) \
          / (l2 * (m1 + m2 * np.sin(o1 - o2) ** 2))
    return o1d, v1d, o2d, v2d


g = 9.81  # Gravitation acceleration
# Angles
o1 = 6.0 * np.pi / 6.0
o2 = 5.0 * np.pi / 8.0
# Velocities
v1 = v2 = 0
# Pendulum params
l1, l2 = 1, 1  # Length
m1, m2 = 1, 1  # Mass

state_0 = np.array([o1, v1, o2, v2])

# Time range (sec)
t = np.arange(0, 60, 1.0/60)

# Calculate results for time range
results = odeint(get_derivative, state_0, t, args=(l1, l2, m1, m2))

# Convert to X, Y coordinates
o1, o2 = results[:, 0], results[:, 2]
x1 = l1 * np.sin(o1)
y1 = -l1 * np.cos(o1)
x2 = x1 + l2 * np.sin(o2)
y2 = y1 - l2 * np.cos(o2)
data = np.array([x1, y1, x2, y2]).transpose()

# Draw results
g, f = create_graph("Double pendulum")
plt.pause(0.1)

for d in data:
    draw_step(g, d, [l1, l2])
    # Redraw
    #plt.pause(0.01)
    f.canvas.draw()
    f.canvas.flush_events()

plt.show()