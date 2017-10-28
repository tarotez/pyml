import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
d = pd.read_csv('GlobalTemperatures.csv')
x = d['LandAndOceanAverageTemperature']
z = np.array([value for value in x if not math.isnan(value)])
bins = np.linspace(z.min(), z.max(), 100)
h = plt.hist(z,bins)
