import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

d = pd.read_csv('winequality-red.csv', sep=';')
indep_var_labels = ['fixed acidity']
indep_targetID = 0
dep_var_label = 'quality'
indep = np.array([d[label] for label in indep_var_labels]).transpose()
quality = d[dep_var_label].values

lr = linear_model.LinearRegression()
lr.fit(indep, quality)

indep_test = indep
quality_predict = lr.predict(indep_test)
fig, ax = plt.subplots()
ax.scatter(indep[:,indep_targetID], quality, c='blue', marker='.', label='observed')
ax.scatter(indep_test[:,indep_targetID], quality_predict, c='red', marker='.', label='prediction')
ax.legend(loc='lower left')

ax.set_xlabel(indep_var_labels[indep_targetID])
ax.set_ylabel(dep_var_label)

error = quality - quality_predict
rmse = np.sqrt(np.mean(error ** 2))
print('rmse = ', rmse)
