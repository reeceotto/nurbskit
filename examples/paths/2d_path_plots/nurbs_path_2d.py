"""
Plotting 2D NURBS curves.

Author: Reece Otto 31/01/2022
"""
from nurbskit.path import NURBS
from nurbskit.visualisation import path_plot_2D
import matplotlib.pyplot as plt

# open 2D NURBS curve
P1 = [[-13, -4], [-10, -3], [-10.2, 1.5], [-7.5, 1.5]]
p1 = 2
U1 = [0, 0, 0, 0.5, 1, 1, 1]
G1 = [1.5, 1.2, 1.7, 1]
nurbs1 = NURBS(P=P1, p=p1, U=U1, G=G1)

# closed 2D NURBS curve
P2 = [[0, 0], [-5, 1.9], [1, 6.9], [5, 0.5], [0, 0]]
p2 = 2
U2 = [0, 0, 0, 0.3, 0.7, 1, 1, 1]
G2 = [1.5, 1.2, 1.7, 1, 1.6]
nurbs2 = NURBS(P=P2, p=p2, U=U2, G=G2)

# create plot
fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 20
        })
ax = path_plot_2D(nurbs1, path_label='Open NURBS', show_knots=False)
ax = path_plot_2D(nurbs2, path_label='Closed NURBS', axes=ax, path_style='r',
        control_point_style='mo', control_net_style='c-')
ax.set_aspect('equal', adjustable="datalim")
plt.grid()
plt.legend()
plt.show()
fig.savefig('nurbs_2d.svg', bbox_inches='tight')
