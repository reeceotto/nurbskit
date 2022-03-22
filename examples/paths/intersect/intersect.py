"""
Finding intersection between two curves.

Author: Reece Otto 21/03/2022
"""
from nurbskit.path import Bezier
from nurbskit.visualisation import path_plot_2D
import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np

# define curves
P1 = [[0, 0], [1, 1], [2, 4], [3, 9], [4, 16]]
bezier1 = Bezier(P=P1)

P2 = [[1, 10], [2, 5], [4, 4], [5, 1], [6, 0]]
bezier2 = Bezier(P=P2)

# find intersection point
def root_func(u, bezier1, bezier2):
	pos_vec = bezier2(u[1]) - bezier1(u[0]) 
	return np.linalg.norm(pos_vec)

bounds = ((0.0, 1.0), (0.0, 1.0))

sol = optimize.minimize(root_func, [0.5, 0.5], method='SLSQP', 
	args=(bezier1, bezier2), bounds=bounds)
point1 = bezier1(sol.x[0])
point2 = bezier2(sol.x[1])
print(sol)

# create plot
fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 20
        })
ax = path_plot_2D(bezier1, show_knots=False, show_control_net=False, 
	show_control_points=False, path_label='Bezier 1')
ax = path_plot_2D(bezier2, path_label='Bezier 2', axes=ax, 
        show_knots=False, show_control_net=False, show_control_points=False)
ax.scatter(point1[0], point1[1], label='Calculated Intersection')
ax.set_aspect('equal', adjustable="datalim")
plt.grid()
plt.legend()
plt.show()
fig.savefig('bezier_intersect.svg', bbox_inches='tight')