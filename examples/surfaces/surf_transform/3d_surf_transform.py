"""
Performing transformations on a 3D NURBS surface.

Author: Reece Otto 14/03/2022
"""
from nurbskit.surface import NURBSSurface
from nurbskit.visualisation import surf_plot_3D
from nurbskit.utils import auto_knot_vector
from nurbskit.geom_utils import scale, translate, rotate_x, rotate_y, rotate_z
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from math import pi

#------------------------------------------------------------------------------#
#                                Original surface                              #
#------------------------------------------------------------------------------#
# generate NURBS surface
P = [[[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [2.0, 0.0, 0.0], [3.0, 0.0, -1.0]],
     [[0.0, 1.0, 1.0], [1.0, 1.0, 0.0], [2.0, 1.0, -1.0], [3.0, 1.0, 0.0]],
     [[0.0, 2.0, 0.0], [1.0, 2.0, -1.0], [2.0, 2.0, 0.0], [3.0, 2.0, 1.0]],
     [[0.0, 3.0, -1.0], [1.0, 3.0, 0.0], [2.0, 3.0, 1.0], [3.0, 3.0, 0.0]]]
G = [[1.0, 2.0, 1.0, 2.0],
     [2.0, 1.0, 2.0, 1.0],
     [1.0, 2.0, 1.0, 2.0],
     [2.0, 1.0, 2.0, 1.0]]
p = 3
q = 3
U = auto_knot_vector(len(P), p)
V = auto_knot_vector(len(P[0]), q)
surf = NURBSSurface(P=P, G=G, p=3, q=3, U=U, V=V)

# create 3D plot
fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({
     "text.usetex": True,
     "font.family": "sans-serif",
     "font.size": 15
     })
ax = surf_plot_3D(surf)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
plt.legend()
plt.show()
fig.savefig('original_surf.svg', bbox_inches='tight')

#------------------------------------------------------------------------------#
#                              Scaled surface                                  #
#------------------------------------------------------------------------------#
P_scaled = scale(np.array(P), x_scale=2.0, y_scale=1.0, z_scale=3.0)
surf_scaled = NURBSSurface(P=P_scaled, G=G, p=3, q=3, U=U, V=V)

# create 3D plot
fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({
     "text.usetex": True,
     "font.family": "sans-serif",
     "font.size": 15
     })
ax = surf_plot_3D(surf_scaled)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
plt.legend()
plt.show()
fig.savefig('scaled_surf.svg', bbox_inches='tight')

#------------------------------------------------------------------------------#
#                           Translated surface                                 #
#------------------------------------------------------------------------------#
P_trans = translate(np.array(P), x_shift=1.0, y_shift=2.0, z_shift=3.5)
surf_trans = NURBSSurface(P=P_trans, G=G, p=3, q=3, U=U, V=V)

# create 3D plot
fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({
     "text.usetex": True,
     "font.family": "sans-serif",
     "font.size": 15
     })
ax = surf_plot_3D(surf_trans)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
plt.legend()
plt.show()
fig.savefig('trans_surf.svg', bbox_inches='tight')

#------------------------------------------------------------------------------#
#                    Surface rotated around x axis                             #
#------------------------------------------------------------------------------#
P_rot_x = rotate_x(np.array(P), 90*pi/180)
surf_rot_x = NURBSSurface(P=P_rot_x, G=G, p=3, q=3, U=U, V=V)

# create 3D plot
fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({
     "text.usetex": True,
     "font.family": "sans-serif",
     "font.size": 15
     })
ax = surf_plot_3D(surf_rot_x)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
plt.legend()
plt.show()
fig.savefig('rot_x_surf.svg', bbox_inches='tight')

#------------------------------------------------------------------------------#
#                    Surface rotated around y axis                             #
#------------------------------------------------------------------------------#
P_rot_y = rotate_y(np.array(P), 90*pi/180)
surf_rot_y = NURBSSurface(P=P_rot_y, G=G, p=3, q=3, U=U, V=V)

# create 3D plot
fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({
     "text.usetex": True,
     "font.family": "sans-serif",
     "font.size": 15
     })
ax = surf_plot_3D(surf_rot_y)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
plt.legend()
plt.show()
fig.savefig('rot_y_surf.svg', bbox_inches='tight')

#------------------------------------------------------------------------------#
#                    Surface rotated around z axis                             #
#------------------------------------------------------------------------------#
P_rot_z = rotate_z(np.array(P), 90*pi/180)
surf_rot_z = NURBSSurface(P=P_rot_z, G=G, p=3, q=3, U=U, V=V)

# create 3D plot
fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({
     "text.usetex": True,
     "font.family": "sans-serif",
     "font.size": 15
     })
ax = surf_plot_3D(surf_rot_z)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
plt.legend()
plt.show()
fig.savefig('rot_z_surf.svg', bbox_inches='tight')