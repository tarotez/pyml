#!/usr/bin/env python
# linear regression to abalone data

from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

# read data
datalist = []
f = open('abalone.data', 'r')
for line in f:
    line = line.rstrip()
    elems = line.split(',')
    datalist.append(elems[1:])
f.close()
data = np.array(datalist)
x = np.c_[np.float_(data[:,0])]
y = np.c_[np.float_(data[:,-1])]

lr = linear_model.LinearRegression()
lr.fit(x, y)

# mesh for x_test
x_test = np.linspace(x.min(), x.max(), 1000)

predicted = lr.coef_[0] * x_test + lr.intercept_ * np.ones(x_test.shape)

fig, ax = plt.subplots()
ax.scatter(x, y, c='red', marker='o', label='observed', lw=0)
ax.scatter(x_test, predicted, c='blue', marker='o', label='predicted', lw=0)

ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=18)
ax.legend(loc='upper left', fontsize=18)
ax.set_xlabel('length', fontsize=18)
ax.set_ylabel('rings', fontsize=18)

plt.show()
