import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
d = pd.read_csv('GlobalTemperatures.csv')
offset = 100
date = d['dt'][offset::12]
year = np.array([int(e.split('-')[0]) for e in date])
temperature = d['LandAverageTemperature'][offset::12]
lr = linear_model.LinearRegression()
lr.fit(year.reshape((-1,1)), temperature.values)
year_test = np.linspace(year.min(), year.max(), 500)
temperature_predict = lr.predict(year_test.reshape((-1,1)))
fig, ax = plt.subplots()
ax.scatter(year, temperature, label='measurement', c='b', marker='.', s=5)
ax.scatter(year_test, temperature_predict, c='red', marker='.', s=10, label='interpolation')
ax.legend(loc='lower right')
ax.set_xlabel('Year')
ax.set_ylabel('Temperature')
