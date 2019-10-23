import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

d = pd.read_csv('winequality-red.csv', sep=';')
indep_var_labels = ['fixed acidity']
dep_var_label = 'quality'
indep = np.array([d[label] for label in indep_var_labels]).transpose()
quality = d[dep_var_label].values

lr = linear_model.LinearRegression()
lr.fit(indep, quality)

indep_test = np.linspace(indep.min(), indep.max(), len(indep))
quality_predict = lr.predict(indep_test.reshape((-1,1)))
fig, ax = plt.subplots()
ax.scatter(indep, quality, label='observed', marker='*')
ax.scatter(indep_test, quality_predict, c='green', marker='.' , label='prediction')
ax.legend(loc='lower left')

targetID = 0
ax.set_xlabel(indep_var_labels[targetID])
ax.set_ylabel(dep_var_label)

error = quality - quality_predict
rmse = np.sqrt(np.mean(error ** 2))
print('rmse = ', rmse)
