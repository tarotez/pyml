#!/usr/bin/env python
# Gaussian process regression to Air Quality Data in UCI-MLR

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

# data read and visualization parameters
offsetTimePoint = 100
maxTimePoint = 96
subsamplingSkip = 6
testTimePointSamplingNum = 1000

# read data
datalist = []
f = open('AirQualityUCI.csv', 'r')
for line in f:
    line = line.rstrip()
    line = line.replace(",",".")
    elems = line.split(';')
    if len(elems[0]) > 1:
        datalist.append(elems)
f.close()
attr_and_data = np.array(datalist)
data = attr_and_data[offsetTimePoint:offsetTimePoint+maxTimePoint:subsamplingSkip,:]
dataMin = 0
dataMax = maxTimePoint
x_train = np.c_[np.linspace(dataMin, dataMax, data.shape[0])]
y_train = np.c_[np.float_(data[:,12])]  # get temperature
x_test = np.c_[np.linspace(dataMin, dataMax, testTimePointSamplingNum)]

# instanciate a Gaussian Process model
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# fit to data using Maximum Likelihood estimation of the parameters
gp.fit(x_train, y_train)

# make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(x_test, return_std=True)

fig, ax = plt.subplots()

ax.fill(np.concatenate([x_test, x_test[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='green', ec='None', label='95% confidence interval', zorder = 0)
ax.plot(x_test, y_pred, label=u'prediction', markersize=30, c='blue', zorder = 1)
ax.scatter(x_train, y_train, c='red', marker='o', label='observed', lw=0, zorder = 2)

ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=18)
ax.set_xlabel('hour', fontsize=18)
ax.set_ylabel('temperature', fontsize=18)

plt.show()
