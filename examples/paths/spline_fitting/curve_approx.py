"""
Fitting a B-Spline path to given point data.

Author: Reece Otto 15/02/2022
"""
from nurbskit.spline_fitting import fit_bspline_path
from nurbskit.visualisation import path_plot_2D
import numpy as np
import matplotlib.pyplot as plt

Q = np.array([[0.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0], [6.0, -1.0]])

spline = fit_bspline_path(Q, 4, 3)

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
plt.savefig('spline_fit.svg', bbox_inches='tight')
plt.show()
