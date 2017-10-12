#!/usr/bin/env python
# logistic regression

from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np

# read data
datalist = []
f = open('../datasets/adult.data', 'r')
sampleNum = 0
for line in f:
    line = line.rstrip()
    elems = line.split(',')
    if len(elems) > 1:
        datalist.append(elems)
        sampleNum += 1
f.close()
dataArray = np.array(datalist)
data = dataArray.reshape(sampleNum,-1)
x = np.c_[np.float_(data[:,0])]
y = data[:,-1]

enc = OneHotEncoder()

x_train = x
y_train = enc.fit(y)
x_test = x
y_test = enc.fit(y)

logr = linear_model.LogisticRegression
logr.fit(x_train, y_train)

# mesh for x_test
x_test = np.linspace(x.min(), x.max(), 1000)

y_predicted = logr(x_test)

fig, ax = plt.subplots()
ax.scatter(x, y, c='red', marker='o', label='observed', lw=0)
ax.scatter(x_test, predicted, c='blue', marker='o', label='predicted', lw=0)

ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=18)
ax.legend(loc='upper left', fontsize=18)
ax.set_xlabel('length', fontsize=18)
ax.set_ylabel('rings', fontsize=18)

plt.show()
