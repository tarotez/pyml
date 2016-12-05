import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
import numpy as np

attrNames = ["sepal length", "sepal width", "petal length", "petal width"]
targetAttr = [0,2]

datalist = []
f = open('iris.data', 'r')
for line in f:
    line = line.rstrip()
    elems = line.split(',')
    if len(elems) > 1:
      datalist.append(elems[:-1])
f.close()

X = np.float_(np.array(datalist))

kres = KMeans(n_clusters=5)
clus_labels = kres.fit_predict(X)

clusCent = kres.cluster_centers_

fig, ax = plt.subplots()
ax.scatter(X[:, targetAttr[0]], X[:, targetAttr[1]], c=clus_labels, lw=0)

ax.scatter(clusCent[:, targetAttr[0]], clusCent[:,targetAttr[1]], edgecolors="red", c="red", lw=10)

ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=18)
ax.set_xlabel(attrNames[targetAttr[0]], fontsize=18)
ax.set_ylabel(attrNames[targetAttr[1]], fontsize=18)

plt.show()
