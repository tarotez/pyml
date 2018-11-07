from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from scipy.stats import multivariate_normal

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(-3, 3, 0.01)
Y = np.arange(-3, 3, 0.01)
X, Y = np.meshgrid(X, Y)

# set parameters
mean = np.array([0,0])
cov = np.array([[1, 0],
                [0, 1]])

Z = multivariate_normal.pdf(np.array([X,Y]).transpose(), mean=mean, cov=cov)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.Reds, linewidth=1, antialiased=False)

# Customize the z axis.
ax.set_zlim(0, 0.3)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
