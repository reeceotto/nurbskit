"""
Plotting 2D Bezier curves.

Author: Reece Otto 31/01/2022
"""
from nurbskit.path import Bezier
from nurbskit.visualisation import path_plot_2D
import matplotlib.pyplot as plt

# open 2D Bezier curve
P1 = [[-13, -4], [-10, -3], [-10.2, 1.5], [-7.5, 1.5]]
bezier1 = Bezier(P=P1)

# closed 2D Bezier curve
P2 = [[0, 0], [-5, 1.9], [1, 6.9], [5, 0.5], [0, 0]]
bezier2 = Bezier(P=P2)

# create plot
fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 20
        })
ax = path_plot_2D(bezier1, show_knots=False, path_label='Open Bezier')
ax = path_plot_2D(bezier2, path_label='Closed Bezier', axes=ax, 
        show_knots=False, path_style='r', control_point_style='mo', 
        control_net_style='c-')
ax.set_aspect('equal', adjustable="datalim")
plt.grid()
plt.legend()
plt.show()
fig.savefig('bezier_2d.svg', bbox_inches='tight')
