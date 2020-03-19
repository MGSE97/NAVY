import numpy as np

from Hoppfield.Hopfield import HopfieldNet
from Hoppfield.HopfieldPatterns import patterns5x5, destroy, prepare_data, convert_data, Pattern, Row

# Prepare network and save patterns
net = HopfieldNet(5, 5)
for p in patterns5x5:
    net.save(prepare_data(p.data))

# Sync repair, stopped by halucination
print("Synchronous repair")
#for p in np.random.choice(patterns5x5, 4):
halucination = False
i = 0
g_success = 0.0
while not halucination:
    # get random pattern
    pattern = np.random.choice(patterns5x5)
    # destroy N parts
    destroyed = destroy(pattern, 5)
    # fix pattern
    fixed = Pattern(convert_data(net.recover(prepare_data(destroyed.data))))
    # compute difference to original
    result = (fixed.data == pattern.data).astype(int)
    # save success rate
    success = np.average(result)
    if success == 1:
        g_success += 1
    # detect halucination
    halucination = not np.any([np.all(fixed.data == s.data) for s in patterns5x5])
    # print results
    print(i, halucination, success,
          "\nDest  Fixed Origo Diff",
          '\n' + str(Row([destroyed, fixed, pattern, Pattern(result)])))
    i += 1

# Async repair, stopped by runtime
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
