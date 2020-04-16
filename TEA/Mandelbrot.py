"""
Drawing Mandelbrot set as graph.
Use Drag select with LEFT mouse click to zoom in.
Use RIGHT mouse click to zoom out.
Use MIDDLE mouse click to reset.
"""

import os

import numpy as np
from matplotlib.backend_bases import MouseButton
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool

Axes3D = Axes3D  # pycharm auto import
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
from matplotlib.widgets import RectangleSelector


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


def draw_points(ax, points):
    """
    Draw points to graph.

    :param ax: graph
    :param positions: position in format [[x,][y,][z,]]
    """
    print("Drawing points...")
    x, y, z = points
    ax.scatter(x, y, c=z, s=1)


def onselect(eclick, erelease):
    """eclick and erelease are matplotlib events at press and release"""
    global rangex
    global rangey
    global zooms
    global f
    global g
    global points
    # Handle clicks
    if eclick.button == MouseButton.LEFT:
        size = f.get_size_inches() * f.dpi
        d = max(abs(erelease.xdata - eclick.xdata), rangex[2]/100)
        erelease.xdata = eclick.xdata + d
        erelease.ydata = eclick.ydata + d/size[0]*size[1]
        # Zoom In
        zooms.append([rangex, rangey])
        rangex, rangey = [
                             eclick.xdata,
                             erelease.xdata,
                             erelease.xdata - eclick.xdata
                         ], [
                             eclick.ydata,
                             erelease.ydata,
                             erelease.ydata - eclick.ydata
                         ]
    elif eclick.button == MouseButton.MIDDLE:
        # Zoom default
        rangex, rangey = [-2, 0.5, 2.5], [-1, 1, 2]
        zooms = [rangex, rangey]
    elif eclick.button == MouseButton.RIGHT:
        # Zoom Out
        if zooms.__len__() > 1:
            rangex, rangey = zooms.pop()

    # Redraw
    print('Range: (%f, %f) => (%f, %f)' % (rangex[0], rangey[0], rangex[1], rangey[1]))
    points = compute(rangex, rangey, max_iterations, m, z0)
    g.clear()
    draw_points(g, points)
    g.set_xlim([rangex[0], rangex[1]])
    g.set_ylim([rangey[0], rangey[1]])


def mandelbrot(args):
    """
    Compute Mandelbrot set at point

    :param args: x, y, z_0, max_iterations
    :return: x, y, color
    """
    x, y, z0, m, max_iterations = args
    z = z0
    n = 0
    c = complex(x, y)
    while abs(z) <= m and n < max_iterations:
        z = z * z + c
        n += 1
    return [x, y, mplc.hsv_to_rgb((n / max_iterations, 1, 1 if n < max_iterations else 0))]


def compute(rangex, rangey, max_iterations, m, z0):
    """
    Compute Mandelbrot set at range

    :param rangex: X range + size
    :param rangey: Y range + size
    :param max_iterations: Iteration limit
    :param m: m
    :param z0: z_0
    :return: points to draw
    """
    print("Computing points...")
    xs = np.arange(rangex[0], rangex[1], rangex[2] / 300)
    ys = np.arange(rangey[0], rangey[1], rangey[2] / 300)
    with Pool(os.cpu_count()) as p:
            points = np.array(p.map(mandelbrot, [(x, y, z0, m, max_iterations) for y in ys for x in xs])).transpose()

    return points


if __name__ == '__main__':
    # Configure
    m = 2
    z0 = 0
    max_iterations = 1000
    rangex, rangey = [-2, 0.5, 2.5], [-1, 1, 2]
    zooms = [rangex, rangey]
    g, f = create_graph("Mandelbrot set")

    # Prepare handlers
    rs = RectangleSelector(g, onselect, drawtype='line')
    rs.set_active(True)

    # First draw
    points = compute(rangex, rangey, max_iterations, m, z0)
    draw_points(g, points)

    # Show
    plt.show()
