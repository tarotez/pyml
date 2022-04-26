import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_path = os.path.join(
    os.path.dirname(__file__), "..", "data", "net_salary_per_town_categories.csv"
)
sal = pd.read_csv(data_path)
x = np.array(sal["SNHMC14"])
bins = np.linspace(x.min(), x.max(), 100)
h = plt.hist(x, bins)
