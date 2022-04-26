import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.mixture

data_path = os.path.join(os.path.dirname(__file__), "..", "data", "Iris.csv")
d = pd.read_csv(data_path)
sl = d["SepalLengthCm"]
pl = d["PetalLengthCm"]
observed = np.transpose(np.array([sl, pl]))
gmm = sklearn.mixture.GaussianMixture(n_components=3)
gmm.fit(observed)
sampleNum = observed.shape[0] * 5
g = gmm.sample(sampleNum)
x = g[0]
z = g[1]
pred_sl = x[:, 0]
pred_pl = x[:, 1]
pred_clus = z
plt.scatter(
    pred_sl[pred_clus == 0], pred_pl[pred_clus == 0], c="green", s=20, edgecolors="none"
)
plt.scatter(
    pred_sl[pred_clus == 1],
    pred_pl[pred_clus == 1],
    c="purple",
    s=20,
    edgecolors="none",
)
plt.scatter(
    pred_sl[pred_clus == 2],
    pred_pl[pred_clus == 2],
    c="orange",
    s=20,
    edgecolors="none",
)
plt.scatter(sl, pl, c="red", marker="x", s=20)
