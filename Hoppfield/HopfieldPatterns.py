import copy

import numpy as np


class Row:
    def __init__(self, patterns):
        self.patterns = patterns

    def __str__(self):
        rows = np.array([str(p).split('\n') for p in self.patterns]).transpose()
        txt = ''
        for row in rows:
            txt += str.join(' ', row) + '\n'
        return txt


class Pattern:
    def __init__(self, data):
        self.data = np.array(data)

    def __str__(self):
        txt = ''
        for row in self.data:
            for item in row:
                txt += str(item)
            txt += '\n'
        return txt


def prepare_data(data):
    """
    Converts 0 to -1 in data

    :param data: Data to convert
    :return: Result
    """
    return np.where(data==0, -1, data)


def convert_data(data):
    """
    Converts -1 to 0 in data

    :param data: Data to convert
    :return: Result
    """
    return np.where(data==-1, 0, data)


def destroy(pattern, count):
    """
    Randomly destroys pattern

    :param pattern: Pattern to destroy
    :param count: Number of changes
    :return: Destroyed pattern
    """
    destroyed = Pattern(copy.deepcopy(pattern.data))
    for i in range(0, count):
        x = int(np.round(np.random.uniform(0, pattern.data.shape[0]-1)))
        y = int(np.round(np.random.uniform(0, pattern.data.shape[1]-1)))
        destroyed.data[y, x] = 0 if destroyed.data[y, x] == 1 else 1
    return destroyed


# Some patterns
patterns5x5 = [
    Pattern([  # 0
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0]
    ]),
    Pattern([  # 1
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0]
    ]),
    Pattern([  # 2
        [0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 1, 1, 0]
    ]),
    Pattern([  # 3
        [0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0]
    ]),
    Pattern([  # 4
        [0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0]
    ]),
    Pattern([  # 5
        [0, 1, 1, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0]
    ]),
    Pattern([  # 6
        [0, 1, 1, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0]
    ]),
    Pattern([  # 7
        [0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ]),
    Pattern([  # 8
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0]
    ]),
    Pattern([  # 9
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0]
    ])
]
