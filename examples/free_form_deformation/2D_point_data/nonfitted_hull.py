"""
Using FFD to deform a discrete path in order to interpolate a set of points.

Author: Reece Otto 04/02/2022
"""
from nurbskit.path import NURBS
from nurbskit.utils import auto_knot_vector
from nurbskit.surface import NURBSSurface
from nurbskit.visualisation import path_plot_2D, surf_plot_2D
from nurbskit.point_inversion import point_inv_surf
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from scipy.optimize import minimize

# define initial path using NURBS
P_curve = [[0, -5], [3, -3], [3.2, 1.5], [7, 2]]
p_curve = 2
U_curve = [0, 0, 0, 0.5, 1, 1, 1]
G_curve = [1.5, 1.2, 1.7, 1]
nurbs_curve = NURBS(P=P_curve, p=p_curve, U=U_curve, G=G_curve)

# intial FFD hull (NURBS surface)
P_hull = [[[-1, -6], [2, -6], [5, -6], [8, -6]],
          [[-1, -3], [2, -3], [5, -3], [8, -3]],
          [[-1, 0],  [2, 0],  [5, 0],  [8, 0]],
          [[-1, 3],  [2, 3],  [5, 3],  [8, 3]]]
G_hull = [[1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1]]
p_hull = 3
q_hull = 3
U_hull = auto_knot_vector(len(P_hull), p_hull)
V_hull = auto_knot_vector(len(P_hull[0]), q_hull)
hull = NURBSSurface(P=P_hull, G=G_hull, p=p_hull, q=q_hull, U=U_hull, V=V_hull)

# point data
Q = np.array([[0, -4], [2, -3], [3, -1], [4, 2], [7, 1]])

# create plot of initial configuration
fig = plt.figure(figsize=(32, 18))
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 18
        })
ax = path_plot_2D(nurbs_curve, n_points=len(Q), label='Initial Path', 
	show_control_points=False, show_control_net=False, show_knots=False)
ax = surf_plot_2D(hull, axes=ax, show_boundary=True,  show_grid=False, 
	show_knots=False, show_weights=True, boundary_label='Initial FFD Hull', 
	control_point_label='Hull Control Points', 
	control_net_label='Hull Control Net', )
ax.plot(Q[:,0], Q[:,1], 'cx', label='Point Data')
plt.legend()
ax.set_aspect('equal', adjustable="datalim")
plt.show()
fig.savefig('nonfitted_config.svg', bbox_inches='tight')

# evalute points along initial nurbs path
path_coords = nurbs_curve.list_eval(n_points=len(Q))

# run point inversion wrt to hull for each path point
params = np.nan * np.ones((len(path_coords), 2))
print('---------- Running point inversion ----------')
for i in range(len(path_coords)):
    params[i] = point_inv_surf(hull, path_coords[i], tol=1E-6)
    print(f'(x, y) = ({path_coords[i][0]:.4}, {path_coords[i][1]:.4}) \
    	==>    (u, v) = ({params[i][0]:.4}, {params[i][1]:.4})')
    if i == len(path_coords) - 1:
        print('')

# define objective function based on the sum of distance_ij
def objective_fcn_1(design_vars, point_data, params, hull):
    P_new = np.reshape(design_vars, hull.P.shape)
    hull_new = NURBSSurface(P=P_new, G=hull.G, p=hull.p, q=hull.q, U=hull.U, 
		V=hull.V)
	
    J = 0
    for i in range(len(point_data)):
	    data_point_i = point_data[i]
	    path_point_i = hull_new(params[i][0], params[i][1])
	    J += sqrt((data_point_i[0] - path_point_i[0])**2 + \
		    	 (data_point_i[1] - path_point_i[1])**2)
    return J

# set the control points of the hull as the design variables and run optimizer
print('---------- Running optimizer ----------')
print('Design variables: control points')
design_vars = hull.P.flatten()
sol = minimize(objective_fcn_1, design_vars, method='SLSQP', 
	args=(Q, params, hull), options={'disp':True})

# extract control point array and coordinates of deformed path
P_sol = np.reshape(sol.x, hull.P.shape)
hull_sol = NURBSSurface(P=P_sol, G=G_hull, p=p_hull, q=q_hull, U=U_hull, 
    V=V_hull)
deformed_path = np.nan * np.ones((len(params), 2))
for i in range(len(params)):
    deformed_path[i] = hull_sol(params[i][0], params[i][1])

# plot new path and perturbed hull
fig = plt.figure(figsize=(32, 18))
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.size": 18
    })
ax = plt.axes()
ax.plot(deformed_path[:,0], deformed_path[:,1], 'k-', label='Optimized Path')
ax = surf_plot_2D(hull_sol, axes=ax, show_boundary=True,  show_grid=False, 
    show_knots=False, show_weights=True, boundary_label='Deformed FFD Hull', 
    control_point_label='Hull Control Points', 
    control_net_label='Hull Control Net', )

ax.plot(Q[:,0], Q[:,1], 'cx', label='Point Data')
plt.legend()
ax.set_aspect('equal', adjustable="datalim")
plt.show()
fig.savefig('nonfitted_opt_Ps.svg', bbox_inches='tight')

# re-run the optimizer, but now add control point weights as design vars
def objective_fcn_2(design_vars, point_data, params, hull):
    no_weights = len(hull.G) * len(hull.G[0])
    G_new = np.reshape(design_vars[-no_weights:], hull.G.shape)
    P_new = np.reshape(design_vars[:-no_weights], hull.P.shape)

    hull_new = NURBSSurface(P=P_new, G=G_new, p=hull.p, q=hull.q, U=hull.U, 
    	V=hull.V)
	
    J = 0
    for i in range(len(point_data)):
	    data_point_i = point_data[i]
	    path_point_i = hull_new(params[i][0], params[i][1])
	    J += sqrt((data_point_i[0] - path_point_i[0])**2 + \
		    (data_point_i[1] - path_point_i[1])**2)
    return J

print('\n---------- Running optimizer ----------')
print('Design variables: control points and weights')
design_vars = np.concatenate((hull.P.flatten(), hull.G.flatten()))
sol = minimize(objective_fcn_2, design_vars, method='SLSQP', 
	args=(Q, params, hull), options={'disp':True})

# extract control point array and coordinates of deformed path
no_weights = len(hull.G) * len(hull.G[0])
G_sol = np.reshape(sol.x[-no_weights:], hull.G.shape)
P_sol = np.reshape(sol.x[:-no_weights], hull.P.shape)
hull_sol = NURBSSurface(P=P_sol, G=G_sol, p=p_hull, q=q_hull, U=U_hull, 
	V=V_hull)
deformed_path = np.nan * np.ones((len(params), 2))
for i in range(len(params)):
    deformed_path[i] = hull_sol(params[i][0], params[i][1])

# plot new path and perturbed hull
fig = plt.figure(figsize=(32, 18))
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.size": 18
    })
ax = plt.axes()
ax.plot(deformed_path[:,0], deformed_path[:,1], 'k-', label='Optimized Path')
ax = surf_plot_2D(hull_sol, axes=ax, show_boundary=True,  show_grid=False, 
    show_knots=False, show_weights=True, boundary_label='Deformed FFD Hull', 
    control_point_label='Hull Control Points', 
    control_net_label='Hull Control Net')

ax.plot(Q[:,0], Q[:,1], 'cx', label='Point Data')
plt.legend()
ax.set_aspect('equal', adjustable="datalim")
plt.show()
fig.savefig('nonfitted_opt_Ps_Gs.svg', bbox_inches='tight')
