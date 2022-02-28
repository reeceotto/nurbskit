"""
Plotting 3D NURBS curves.

Author: Reece Otto 31/01/2022
"""
from nurbskit.path import NURBS
from nurbskit.visualisation import path_plot_3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# open 3D NURBS curve
P1 = [[-8, -4, 1], [-5, -3, -3], [-6.2, 1.5, 1], [-4.5, 1.5, -3]]
p1 = 2
U1 = [0, 0, 0, 0.5, 1, 1, 1]
G1 = [1.5, 1.2, 1.7, 1]
nurbs1 = NURBS(P=P1, p=p1, U=U1, G=G1)

# closed 3D bezier curve
P2 = [[0, 0, -2], [-5, 1.9, 2], [1, 6.9, -2], [5, 0.5, 2], [0, 0, -2]]
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
ax = path_plot_3D(nurbs1, path_label='Open NURBS', show_knots=False)
ax = path_plot_3D(nurbs2, path_label='Closed NURBS', axes=ax, path_style='r',
        control_point_style='mo', control_net_style='c-')
plt.legend()
plt.show()
fig.savefig('nurbs_3d.svg', bbox_inches='tight')
