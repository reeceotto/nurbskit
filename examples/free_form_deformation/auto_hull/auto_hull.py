"""
Testing the auto hull functions used in free-form deformation.

Author: Reece Otto 10/02/2022
"""
from nurbskit.ffd_utils import auto_hull_2D
from nurbskit.visualisation import surf_plot_2D
import matplotlib.pyplot as plt
import numpy as np

Q = np.array([[0, -5], [3, -3], [3.2, 1.5], [7, 2]])
hull = auto_hull_2D(Q, 3, 4, p=2)

fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 16
        })
ax = surf_plot_2D(hull, show_knots=False)
ax.plot(Q[:,0], Q[:,1], 'cx', label='Point Data')
ax.set_aspect('equal', adjustable="datalim")
plt.legend()
plt.show()
fig.savefig('auto_hull_2D.svg', bbox_inches='tight')
