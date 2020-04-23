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
    ax = fig.gca(projection='3d')
    fig.canvas.draw_idle()
    return ax, fig


def draw_points(ax, points, cmap):
    """
    Draw points to graph as surface.

    :param ax: graph
    :param positions: position in format [[x,][y,][z,]]
    """
    print("Drawing points...")
    x, y, z = points.T
    ax.plot_trisurf(x, y, z, cmap=cmap, antialiased=True, vmin=0, vmax=1)


def create_surface(size):
    """
    Create surface of size

    :param size: Surface size (width, height)
    :return: Surface points
    """
    w, h = size
    surface = []
    iw, ih = 1 / w, 1 / h
    for x in np.arange(0, 1+iw, iw):
        for y in np.arange(0, 1+ih, ih):
            z = np.random.uniform(0, 1)
            surface.append([x, y, z])

    return np.array(surface)

def clamp_surface(surface, min, max=-1):
    """
    Clamp surfece between min and max value

    :param surface: Surface points
    :param min: Min value or -1 to ignore
    :param max: Max value or -1 to ignore
    """
    for point in surface:
        if min > -1 and point[2] < min:
            point[2] = min
        if max > -1 and point[2] > max:
            point[2] = max


def point_between(a, b):
    """
    Get point between points

    :param a: Point A
    :param b: Point B
    :return: A + (B - A) / 2
    """
    return np.add(a, np.subtract(b, a) / 2).flatten()


def get_point(arr, x, y, w, h):
    """
    Get point from array based on x, y value.
    Array has to be sorted!

    :param arr: Array
    :param x: X value
    :param y: Y value
    :param w: Width
    :param h: Height
    :return: Point [x,y,z]
    """
    # boundary check
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x >= 1:
        x = 1
    if y >= 1:
        y = 1

    # non-sorted variant (slow)
    '''item = []
    for i in arr:
        if i[0] == x and i[1] == y:
            item = i
            break

    return item'''

    # find index and return
    i = int(x*w*(h+1)+y*h)
    return arr[i]


def scale_up_surface(surface, size, iteration=1, roughness=2):
    """
    Scale-up 2x surface point density.

    :param surface: Points array sorted
    :param size: Current size
    :param iteration: This iteration
    :return: [scaled surface, new size, next iteration]
    """
    new_surface = []
    w, h = size
    dw, dh = 1 / w, 1 / h
    scale = 1 / (roughness**iteration)

    # Scale-up
    for x in np.arange(0, 1+dw, dw):
        for y in np.arange(0, 1+dh, dh):

            # 0, 0 point
            if x <= 0 and y <= 0:
                # 0, 0
                new_surface.append(get_point(surface, x, y, w, h))

            # X = 0 axis
            elif x <= 0 < y:
                t, b = get_point(surface, x, y-dh, w, h), get_point(surface, x, y, w, h)
                # tl - tr
                tb = point_between(t, b)
                tb[2] += np.random.uniform(-scale, scale)
                new_surface.append(tb)
                # b
                new_surface.append(b)

            # Y = 0 axis
            elif x > 0 >= y:
                l, r = get_point(surface, x-dw, y, w, h), get_point(surface, x, y, w, h)
                # tl - tr
                lr = point_between(l, r)
                lr[2] += np.random.uniform(-scale, scale)
                new_surface.append(lr)
                # r
                new_surface.append(r)

            # X && Y > 0
            else:
                tl, tr, bl, br = get_point(surface, x-dw, y-dh, w, h), get_point(surface, x, y-dh, w, h), get_point(surface, x-dw, y, w, h), get_point(surface, x, y, w, h)
                # tr - br
                trbr = point_between(tr, br)
                trbr[2] += np.random.uniform(-scale, scale)
                new_surface.append(trbr)
                # tl - br
                tlbr = point_between(tl, br)
                tlbr[2] += np.random.uniform(-scale, scale)
                new_surface.append(tlbr)
                # bl - br
                blbr = point_between(bl, br)
                blbr[2] += np.random.uniform(-scale, scale)
                new_surface.append(blbr)
                # br
                new_surface.append(br)


    # Info data
    print("({:d}, {:d}) => ({:d}, {:d}): {:d} / {:d}".format(w, h, w*2, h*2, new_surface.__len__(), (w*2+1)*(h*2+1)))
    # Sort new surface
    new_surface = np.array(new_surface)
    new_surface = np.array(new_surface[np.lexsort((new_surface[:, 1], new_surface[:, 0]))])
    #print(new_surface)

    # Return new surface, new size, next iteration
    return [new_surface, (w*2, h*2), iteration+1]


# Create graph
g, f = create_graph("Surface")

# Surface configuration
seed = 45           # surface seed
size = (2, 2)       # initial size
scale_to = 64       # scaling size
roughness = 2.5     # terrain roughtness <rough = 1, smooth = 4>
sea_lvl = 0.2       # sea level clamp

# Initial surface
np.random.seed(seed)
surface = create_surface(size)

# Increase size
i = 1
while size[0] < scale_to:
    surface, size, i = scale_up_surface(surface, size, i, roughness)

# Apply sea level
clamp_surface(surface, sea_lvl)

# Draw result
draw_points(g, surface, 'terrain')

plt.show()

