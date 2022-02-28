"""
Testing point projection function for a NURBS path.

Author: Reece Otto 15/02/2022
"""
from nurbskit.point_inversion import point_proj_path
from nurbskit.path import NURBS
from nurbskit.visualisation import path_plot_2D
import matplotlib.pyplot as plt
import numpy as np

# 2D NURBS curve
P = [[-13, -4], [-10, -3], [-10.2, 1.5], [-7.5, 1.5]]
p = 2
U = [0, 0, 0, 0.5, 1, 1, 1]
G = [1.5, 1.2, 1.7, 1]
path = NURBS(P=P, p=p, U=U, G=G)

# define array of points to be projected onto path
Q = np.array([[-15, -4], [-11, -3.5], [-9, -1], [-9, 1], [-7, 2], [-10, 1.5],
	[-11, 0], [-11, -2], [-12.5, -3]])

# run point projection algorithm
proj_coords = np.nan * np.ones(Q.shape)
for i in range(len(Q)):
	xyz_point = Q[i]
	print(f'Projecting point {i}.')
	u_calc = point_proj_path(path, xyz_point, tol_dist=1E-6, tol_cos=1E-6, tol_end=1E-12)
	
	proj_coords[i] = path(u_calc)

# plot path, point data and projected points
plt.figure(figsize=(16, 9))
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 20
        })
ax = path_plot_2D(path, show_knots=False, show_control_points=False, 
	show_control_net=False, path_label='Path')
ax.scatter(Q[:,0], Q[:,1], color='red', label='Point Data')
for i in range(len(Q)):
	if i == 0:
		ax.plot([Q[i][0], proj_coords[i][0]], [Q[i][1], proj_coords[i][1]], 
			'b--', label='Projection Line')
	else:
		ax.plot([Q[i][0], proj_coords[i][0]], [Q[i][1], proj_coords[i][1]], 
			'b--')
ax.scatter(proj_coords[:,0], proj_coords[:,1], color='green', 
	label='Projected Points')
plt.axis('equal')
plt.grid()
plt.legend()
plt.savefig('path_proj.svg', bbox_inches='tight')
plt.show()
