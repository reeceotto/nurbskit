"""
Plotting a 2D NURBS surface.

Author: Reece Otto 07/10/2021
"""
from nurbskit.surface import NURBSSurface
from nurbskit.utils import auto_knot_vector
from nurbskit.visualisation import surf_plot_2D
import matplotlib.pyplot as plt

P = [[[-2, 1], [-1, 2], [0, 1], [1, 2], [2, 1]],
     [[-2, 0], [-1, 0], [0, 0], [1, 0], [2, 0]],
     [[-2, -1], [-1, -2], [0, -1], [1, -2], [2, -1]]]
G = [[2, 1, 2, 1, 2],
     [1, 2, 1, 2, 1],
     [2, 1, 2, 1, 2]]
p = 2
q = 3
U = auto_knot_vector(len(P), p)
V = auto_knot_vector(len(P[0]), q)
nurbs = NURBSSurface(P=P, G=G, p=p, q=q, U=U, V=V)

fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 16
        })
ax = surf_plot_2D(nurbs, show_grid=False, show_boundary=True)
ax.set_aspect('equal', adjustable="datalim")
plt.legend()
fig.show()
fig.savefig('nurbs_surf_2d.svg', bbox_inches='tight')
