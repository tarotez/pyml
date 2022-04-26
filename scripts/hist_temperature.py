import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_path = os.path.join(
    os.path.dirname(__file__), "..", "data", "GlobalTemperatures.csv"
)
d = pd.read_csv(data_path)
x = d["LandAndOceanAverageTemperature"]
z = np.array([value for value in x if not math.isnan(value)])
bins = np.linspace(z.min(), z.max(), 100)
h = plt.hist(z, bins)
