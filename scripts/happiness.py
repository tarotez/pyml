import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

data_path = os.path.join(os.path.dirname(__file__), "..", "data", "2017.csv")
d = pd.read_csv(data_path)
happiness = np.array(d["Happiness.Score"])
gdp = np.array(d["Economy..GDP.per.Capita."])
family = np.array(d["Family"])
plt.scatter(gdp, happiness)
plt.xlabel("GDP", fontsize=18)
plt.ylabel("Happiness", fontsize=18)
