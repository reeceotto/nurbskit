"""
Plotting 2D B-Spline curves.

Author: Reece Otto 31/01/2022
"""
from nurbskit.path import BSpline
from nurbskit.visualisation import path_plot_2D
import matplotlib.pyplot as plt

# open 2D B-Spline curve
P1 = [[-13, -4], [-10, -3], [-10.2, 1.5], [-7.5, 1.5]]
p1 = 2
U1 = [0, 0, 0, 0.5, 1, 1, 1]
spline1 = BSpline(P=P1, p=p1, U=U1)

# closed 2D B-Spline curve
P2 = [[0, 0], [-5, 1.9], [1, 6.9], [5, 0.5], [0, 0]]
p2 = 2
U2 = [0, 0, 0, 0.3, 0.7, 1, 1, 1]
spline2 = BSpline(P=P2, p=p2, U=U2)

# create plot
fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 20
        })
ax = path_plot_2D(spline1, show_knots=False, path_label='Open B-Spline')
ax = path_plot_2D(spline2, path_label='Closed B-Spline', axes=ax, 
        path_style='r',control_point_style='mo', 
        control_net_style='c-')
ax.set_aspect('equal', adjustable="datalim")
plt.grid()
plt.legend()
plt.show()
fig.savefig('bspline_2d.svg', bbox_inches='tight')
