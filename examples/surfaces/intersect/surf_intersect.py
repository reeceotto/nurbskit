"""
Finding the intersection between two 3D surfaces

Author: Reece Otto 21/03/2022
"""
from nurbskit.surface import NURBSSurface
from nurbskit.visualisation import surf_plot_3D
from nurbskit.utils import auto_knot_vector
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import optimize

# generate NURBS surfaces
P1 = [[[0, 0, 4], [1, 0, 1], [2, 0, 0], [3, 0, 1], [4, 0, 4]],
     [[0, 1, 4], [1, 1, 1], [2, 1, 0], [3, 1, 1], [4, 1, 4]],
     [[0, 2, 4], [1, 2, 1], [2, 2, 0], [3, 2, 1], [4, 2, 4]],
     [[0, 3, 4], [1, 3, 1], [2, 3, 0], [3, 3, 1], [4, 3, 4]]]
G1 = [[1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1]]
p1 = 3
q1 = 3
U1 = auto_knot_vector(len(P1), p1)
V1 = auto_knot_vector(len(P1[0]), q1)
surf1 = NURBSSurface(P=P1, G=G1, p=p1, q=q1, U=U1, V=V1)

P2 = [[[0, 0, 5], [1, 0, 3], [2, 0, 2.3], [3, 0, 1.4], [4, 0, 0.4]],
     [[0, 1, 5], [1, 1, 3], [2, 1, 2.1], [3, 1, 1.1], [4, 1, 0.1]],
     [[0, 2, 5], [1, 2, 3], [2, 2, 2.1], [3, 2, 1.1], [4, 2, 0.1]],
     [[0, 3, 5], [1, 3, 3], [2, 3, 2.3], [3, 3, 1.4], [4, 3, 0.4]]]
G2 = [[1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1]]
p2 = 3
q2 = 3
U2 = auto_knot_vector(len(P2), p2)
V2 = auto_knot_vector(len(P2[0]), q2)
surf2 = NURBSSurface(P=P2, G=G2, p=p2, q=q2, U=U2, V=V2)

# find intersection curve
def func(x, surf1, surf2):
	pos_vec = surf2(x[2], x[3]) - surf1(x[0], x[1])
	return np.linalg.norm(pos_vec)

lower_bnd = [0.0, 0.0, 0.0, 0.0]
upper_bnd = [0.99, 0.99, 0.99, 0.99]
bounds = optimize.Bounds(lower_bnd, upper_bnd, keep_feasible=True)
points = []
tol = 1E-4
n_points = 4
dx = 1/(n_points-1)
for i in range(n_points):
	for j in range(n_points):
		for k in range(n_points):
			for l in range(n_points):
				sol = optimize.minimize(func, [i*dx, j*dx, k*dx, l*dx], method='SLSQP', 
					args=(surf1, surf2), bounds=bounds, options={'ftol':1E-8})
				if sol.success and sol.fun <= tol:
					points.append(surf1(sol.x[0], sol.x[1]))
points = np.array(points)
print(f'Number of points found: {len(points)}')
# create 3D plot
fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({
     "text.usetex": True,
     "font.family": "sans-serif",
     "font.size": 15
     })
ax = surf_plot_3D(surf1, show_wireframe=True, 
	show_knots=False, show_control_net=False, show_control_points=False)
ax = surf_plot_3D(surf2, axes=ax, show_wireframe=True,
	show_knots=False, show_control_net=False, show_control_points=False)
#ax.plot(points[:,0], points[:,1], points[:,2], label='Calculated Intersection')
ax.scatter(points[:,0], points[:,1], points[:,2])
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
plt.legend()
plt.show()
fig.savefig('surf_intersect.svg', bbox_inches='tight')