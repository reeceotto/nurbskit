"""
Plotting 3D NURBS surfaces.

Author: Reece Otto 03/02/2022
"""
from nurbskit.surface import NURBSSurface
from nurbskit.visualisation import surf_plot_3D
from nurbskit.utils import auto_knot_vector
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# generate NURBS surface
P = [[[0, 0, 0], [1, 0, 1], [2, 0, 0], [3, 0, -1]],
     [[0, 1, 1], [1, 1, 0], [2, 1, -1], [3, 1, 0]],
     [[0, 2, 0], [1, 2, -1], [2, 2, 0], [3, 2, 1]],
     [[0, 3, -1], [1, 3, 0], [2, 3, 1], [3, 3, 0]]]
G = [[1, 2, 1, 2],
     [2, 1, 2, 1],
     [1, 2, 1, 2],
     [2, 1, 2, 1]]
p = 3
q = 3
U = auto_knot_vector(len(P), p)
V = auto_knot_vector(len(P[0]), q)
surf = NURBSSurface(P=P, G=G, p=3, q=3, U=U, V=V)

# create 3D plot
fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({
     "text.usetex": True,
     "font.family": "sans-serif",
     "font.size": 15
     })
ax = surf_plot_3D(surf)
P_flat = np.array(P).reshape((len(P)*len(P[0]), 3))
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
plt.legend()
plt.show()
fig.savefig('nurbs_surf_3d.svg', bbox_inches='tight')