import numpy as np
import matplotlib.pyplot as plt

clusterNum = 3
iterationNum = 5

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
samples = X[:,targetAttr]
minVec = np.min(X,axis=0)
maxVec = np.max(X,axis=0)

def initializeClusters(samples, clusterNum):
    maxVec = np.max(samples,axis=0)
    minVec = np.min(samples,axis=0)
    clusterCenters = []
    for clusterID in range(clusterNum):
        clusterCenters.append(np.array([np.random.uniform(minVal,maxVal) for minVal, maxVal in zip(minVec, maxVec)]))
    return clusterCenters

def updateClusterCenters(samples, assignments, clusterNum):
    clusterCenters = []
    for clusterID in range(clusterNum):
        sampleList = []
        for sampleID, sample in enumerate(samples):
            if assignments[sampleID] == clusterID:
                sampleList.append(sample)
        clusterCenters.append(np.mean(np.array(sampleList), axis=0))
    return clusterCenters

def assignToClosest(samples, clusterCenters):
    assignments = []
    for sample in samples:
        closestClusterID = np.argmin([np.linalg.norm(sample - clusterCenter) for clusterCenter in clusterCenters])
        assignments.append(closestClusterID)
    return assignments

def drawScatter(samples, assignments, clusterCenters, title):
    fig, ax = plt.subplots()
    ax.scatter(samples[:,0], samples[:,1], c=assignments, lw=0)
    clusterCentersArray = np.array(clusterCenters)
    clusterNum = len(clusterCenters)
    ax.scatter(clusterCentersArray[:,0], clusterCentersArray[:,1], edgecolors="red", c=[i for i in range(clusterNum)], s=300, lw=3)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.set_title(title)
    ax.set_xlabel(attrNames[targetAttr[0]], fontsize=18)
    ax.set_ylabel(attrNames[targetAttr[1]], fontsize=18)
    plt.show()

clusterCenters = initializeClusters(samples, clusterNum)
initial_assignments = np.array([0 for i in range(samples.shape[0])])
drawScatter(samples, initial_assignments, clusterCenters, 'Iteration 0, randomly placed clusters.')
for i in range(iterationNum):
    assignments = assignToClosest(samples, clusterCenters)
    drawScatter(samples, assignments, clusterCenters, 'Iterattion ' + str(i+1) + ', after assignment of labels.')
    clusterCenters = updateClusterCenters(samples, assignments, clusterNum)
    drawScatter(samples, assignments, clusterCenters, 'Iteration ' + str(i+1) + ', after updating cluster locations.')
