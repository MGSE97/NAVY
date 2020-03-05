import numpy as np

from Hopfield import HopfieldNet
from HopfieldPatterns import patterns5x5, destroy, prepare_data, convert_data, Pattern, Row

net = HopfieldNet(5, 5)
for p in patterns5x5:
    net.save(prepare_data(p.data))

#for p in np.random.choice(patterns5x5, 4):
h = False
i = 0
while(not h):
    p = np.random.choice(patterns5x5)
    d = destroy(p, 5)
    f = Pattern(convert_data(net.recover(prepare_data(d.data))))
    r = (f.data == p.data).astype(int)
    h = not np.any([np.all(f.data == s.data) for s in patterns5x5])
    print(i, h, '\n' + str(Row([d, f, p, Pattern(r)])))
    i += 1

