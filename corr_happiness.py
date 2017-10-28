import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
d = pd.read_csv('2017.csv')
happiness = np.array(d['Happiness.Score'])
gdp = np.array(d['Economy..GDP.per.Capita.'])
family = np.array(d['Family'])
plt.scatter(gdp,happiness)
