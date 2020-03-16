import numpy as np

from Hopfield import HopfieldNet
from HopfieldPatterns import patterns5x5, destroy, prepare_data, convert_data, Pattern, Row

net = HopfieldNet(5, 5)
for p in patterns5x5:
    net.save(prepare_data(p.data))

print("Synchronous repair")
#for p in np.random.choice(patterns5x5, 4):
halucination = False
i = 0
g_success = 0.0
while not halucination:
    pattern = np.random.choice(patterns5x5)
    destroyed = destroy(pattern, 5)
    fixed = Pattern(convert_data(net.recover(prepare_data(destroyed.data))))
    result = (fixed.data == pattern.data).astype(int)
    success = np.average(result)
    if success == 1:
        g_success += 1
    halucination = not np.any([np.all(fixed.data == s.data) for s in patterns5x5])
    print(i, halucination, success,
          "\nDest  Fixed Origo Diff",
          '\n' + str(Row([destroyed, fixed, pattern, Pattern(result)])))
    i += 1

print("Asynchronous repair")
halucination = False
j = 0
g_success2 = 0.0
while not halucination and j < 100:
    pattern = np.random.choice(patterns5x5)
    destroyed = destroy(pattern, 5)
    fixed = Pattern(convert_data(net.recover(prepare_data(destroyed.data), False)))
    result = (fixed.data == pattern.data).astype(int)
    success = np.average(result)
    if success == 1:
        g_success2 += 1
    halucination = not np.any([np.all(fixed.data == s.data) for s in patterns5x5])
    print(j, halucination, success,
          "\nDest  Fixed Origo Diff",
          '\n' + str(Row([destroyed, fixed, pattern, Pattern(result)])))
    j += 1

print("Average success Sync:  {:.2%}".format(g_success/i))
print("Average success Async: {:.2%}".format(g_success2/j))
print("Halucinations Sync: {}, Async: {}".format(i, j))
