#!/usr/bin/env python
# polynomial regression to abalone data

from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

datalist = []
f = open('abalone.data', 'r')
for line in f:
    line = line.rstrip()
    elems = line.split(',')
    datalist.append(elems[1:])
f.close()
data = np.array(datalist)
x = np.c_[np.float_(data[:,0])]
y = np.c_[np.float_(data[:,4])]

lr = linear_model.LinearRegression()

X = np.hstack((x, np.power(x,2)))

lr.fit(X, y)

predicted = lr.coef_[0,1] * np.power(x,2) + lr.coef_[0,0] * x + lr.intercept_ * np.ones(x.shape)

fig, ax = plt.subplots()
ax.scatter(x, y, c='blue', marker='o', label='observed', lw=0)
ax.scatter(x, predicted, c='red', marker='o', label='predicted', lw=0)

ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=18)
ax.legend(loc='upper left')
ax.set_xlabel('length', fontsize=18)
ax.set_ylabel('whole weight', fontsize=18)

plt.show()
