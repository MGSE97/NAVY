import numpy as np
import matplotlib.pyplot as plt


class Point:
    data = [0, 0]
    label = 0

    def rand(self, min_val, max_val):
        self.data = np.random.uniform(min_val, max_val, 2)
        return self

    def set(self, data, label):
        self.data = data
        self.label = label
        return self


def rand_point_arr(size, min_val=-100, max_val=100):
    return [Point().rand(min_val, max_val) for x in range(0, size)]


def prepareView(name):
    fig = plt.figure(name)
    fig.suptitle(name)
    ax = fig.gca()
    fig.canvas.draw_idle()
    return ax
