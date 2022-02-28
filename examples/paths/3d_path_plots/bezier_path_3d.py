"""
Plotting 3D Bezier curves.

Author: Reece Otto 31/01/2022
"""
from nurbskit.path import Bezier
from nurbskit.visualisation import path_plot_3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# open 3D Bezier curve
P1 = [[-8, -4, 1], [-5, -3, -3], [-6.2, 1.5, 1], [-4.5, 1.5, -3]]
bezier1 = Bezier(P=P1)

# closed 3D Bezier curve
P2 = [[0, 0, -2], [-5, 1.9, 2], [1, 6.9, -2], [5, 0.5, 2], [0, 0, -2]]
bezier2 = Bezier(P=P2)

# create plot
fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 20
        })
ax = path_plot_3D(bezier1, path_label='Open Bezier', show_knots=False)
ax = path_plot_3D(bezier2, path_label='Closed Bezier', axes=ax, 
        show_knots=False, path_style='r',control_point_style='mo', 
        control_net_style='c-')
plt.legend()
plt.show()
fig.savefig('bezier_3d.svg', bbox_inches='tight')
