import numpy as np
import matplotlib.pyplot as plt
sampleNum = 10000
timeLength = 100
binNum = 50
loc = np.zeros((sampleNum))
for i in range(sampleNum):
    x = 0
    for t in range(timeLength):
        if np.random.randint(2):
            x += 1
        else:
            x -= 1
    loc[i] = x
stride = np.ceil((loc.max() - loc.min()) / binNum)
bins = np.array(range(np.int(loc.min()), np.int(loc.max()), np.int(stride)))
h = plt.hist(loc, bins)
