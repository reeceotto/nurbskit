"""
Global surface interpolation example.

Author: Reece Otto 02/12/2021
"""
from nurbskit.spline_fitting import global_surf_interp
from nurbskit.surface import BSplineSurface
from nurbskit.visualisation import surf_plot_3D
import matplotlib.pyplot as plt
import numpy as np

# define point data to be interpolated
Q = [[[0, 1, 0], [1, 1, 0], [1, 0, 0], [1, -1, 0], [0, -1, 0], [-1, -1, 0], 
      [-1, 0, 0], [-1, 1, 0], [0, 1, 0]],
     [[0, 1, 1], [1, 1, 1], [1, 0, 1], [1, -1, 1], [0, -1, 1], [-1, -1, 1], 
      [-1, 0, 1], [-1, 1, 1], [0, 1, 1]],
     [[0, 1, 2], [1, 1, 2], [1, 0, 2], [1, -1, 2], [0, -1, 2], [-1, -1, 2], 
      [-1, 0, 2], [-1, 1, 2], [0, 1, 2]]]

# calculate interpolative B-spline Surface
p = 2
q = 3
U, V, P = global_surf_interp(Q, p, q)
spline_patch = BSplineSurface(p=p, q=q, U=U, V=V, P=P)

# create 3D plot
fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({
     "text.usetex": True,
     "font.family": "sans-serif",
     "font.size": 15
     })
ax = surf_plot_3D(spline_patch, show_knots=False, N_v=40)
Q_flat = np.array(Q).reshape((len(Q)*len(Q[0]), 3))
ax.scatter(Q_flat[:,0], Q_flat[:,1], Q_flat[:,2], color='blue', 
      label='Point Data')
plt.legend()
plt.show()