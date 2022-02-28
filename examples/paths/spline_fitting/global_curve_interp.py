"""
Global curve interpolation example (pg. 367 of The NURBS book).

Author: Reece Otto 02/12/2021
"""
from nurbskit.path import BSpline
from nurbskit.spline_fitting import global_curve_interp
from nurbskit.visualisation import path_plot_2D
import numpy as np
import matplotlib.pyplot as plt

Q = np.array([[0.0, 0.0], [3.0, 4.0], [-1.0, 4.0], [-4.0, 0.0], [-4.0, -3.0]])
p = 3
U, P = global_curve_interp(Q, p)
spline = BSpline(P=P, U=U, p=p)

# plot points and interpolative B-Spline curve
plt.figure(figsize=(16, 9))
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 20
        })
ax = path_plot_2D(spline, show_knots=False, path_label='Interpolative B-Spline')
ax.scatter(Q[:,0], Q[:,1], label='Point Data')
plt.axis('equal')
plt.grid()
plt.legend()
plt.savefig('global_curve_interp.svg', bbox_inches='tight')
plt.show()
