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
            loc[i] += 1
    x *= 2
    x -= timeLength
    x = loc[i] 
stride = np.int(np.ceil((loc.max() - loc.min()) / binNum))
bins = np.array(range(np.int(loc.min()), np.int(loc.max()), stride))
h = plt.hist(loc, bins)