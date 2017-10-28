import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
d = pd.read_csv('2017.csv')
happiness = d['Happiness.Score']
gdp = d['Economy..GDP.per.Capita.']
japanID = 50
jhappiness = d.ix[japanID,'Happiness.Score']
jgdp = d.ix[japanID, 'Economy..GDP.per.Capita.']
lr = linear_model.LinearRegression()
lr.fit(gdp.values.reshape((-1,1)), happiness.values)
gdp_test = np.linspace(gdp.min(), gdp.max(), 10000)
happiness_predict = lr.predict(gdp_test.reshape((-1,1)))
fig, ax = plt.subplots()
ax.scatter(gdp, happiness, label='All', marker='*')
ax.scatter(jgdp, jhappiness, c='red', label='Japan', s=50)
ax.scatter(gdp_test, happiness_predict, c='green', marker='.' , label='pred')
ax.legend(loc='lower left')
ax.set_xlabel('Economy..GDP.per.Capita.')
ax.set_ylabel('Happiness.Score')