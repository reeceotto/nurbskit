"""
Plotting 3D B-Spline curves.

Author: Reece Otto 31/01/2022
"""
from nurbskit.path import BSpline
from nurbskit.visualisation import path_plot_3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# open 3D B-Spline curve
P1 = [[-8, -4, 1], [-5, -3, -3], [-6.2, 1.5, 1], [-4.5, 1.5, -3]]
p1 = 2
U1 = [0, 0, 0, 0.5, 1, 1, 1]
spline1 = BSpline(P=P1, p=p1, U=U1)

# closed 3D bezier curve
P2 = [[0, 0, -2], [-5, 1.9, 2], [1, 6.9, -2], [5, 0.5, 2], [0, 0, -2]]
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
ax = path_plot_3D(spline1, path_label='Open B-Spline', show_knots=False)
ax = path_plot_3D(spline2, path_label='Closed B-Spline', axes=ax, 
        path_style='r', control_point_style='mo', control_net_style='c-')
plt.legend()
plt.show()
fig.savefig('bspline_3d.svg', bbox_inches='tight')
