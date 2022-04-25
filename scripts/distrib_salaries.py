import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sal = pd.read_csv("net_salary_per_town_categories.csv")
x = np.array(sal["SNHMC14"])
bins = np.linspace(x.min(), x.max(), 100)
h = plt.hist(x, bins)
