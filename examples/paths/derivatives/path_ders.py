"""
Comparing analytical path derivatives to finite difference approximations.

Author: Reece Otto 17/10/2021
"""

from nurbskit.path import BSpline, NURBS
from nurbskit.utils import ders_basis_funs, basis_funs, one_basis_fun, \
                        ders_one_basis_fun, find_span
import numpy as np

# B-Spline basis functions (example from pg. 68 of The NURBS Book)
p = 2
U = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]
u = 5/2
i = 4
n = 2
print("----- Basis function tests -----")
print("Basis function N_{4,2}(5/2):")
print('answer from P&T:', 1/8)
print('using one_basis_fun:', one_basis_fun(i, u, p, U))
print('using basis_funs:', basis_funs(i, u, p, U)[-1])
print('using ders_basis_funs:', ders_basis_funs(i, u, p, n, U)[0][-1])
print('using ders_one_basis_fun:', ders_one_basis_fun(i, u, p, n, U)[0], '\n')

print("Basis function derivative N'_{4,2}(5/2):")
print('answer from P&T:', 0.5)
print('using ders_basis_funs:', ders_basis_funs(i, u, p, n, U)[1][-1])
print('using ders_one_basis_fun:', ders_one_basis_fun(i, u, p, n, U)[1], '\n')

print("Basis function double derivative N''_{4,2}(5/2):")
print('answer from P&T:', 1.0)
print('using ders_basis_funs:', ders_basis_funs(i, u, p, n, U)[2][-1])
print('using ders_one_basis_fun:', ders_one_basis_fun(i, u, p, n, U)[2], '\n')

# B-Spline curve (example from pg. 68, 91 of The NURBS Book)
P = [[-4, -4, 1], [-2, 4, 0], [2, -4, -1], [4, 4, 0], [-1, 1, -5], [1, 1, -3],
     [3, 1, 0], [5, 2, -1]]
p = 2
U = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]
spline = BSpline(P=P, p=p, U=U)
u = 5/2
I = 3

print("----- B-Spline path tests -----")
print('Curve evaluated at u: C(u)')
print('answer from P&T:', (1/8) * np.array(P[2]) + (6/8) * np.array(P[3]) + \
                          (1/8) * np.array(P[4]))
print('using point evaluation:', spline(u))
print('using ders function:', spline.ders(u, 2)[0], '\n')

print("First derivative of curve at u: C'(u)")
print('answer from P&T:', -0.5 * np.array(P[2]) + 0.5 * np.array(P[4]))
print('using finite differences:', spline.dCdu_fd(u))
print('using ders function:', spline.ders(u, 2)[1], '\n')

print("Second derivative of curve at u: C''(u)")
print('using finite differences:', spline.d2Cdu2_fd(u))
print('using ders function:', spline.ders(u, 2)[2], '\n')

print("Derivative wrt control point Px_{I}: dC/dPx_{I}")
print('using finite differences:', spline.dCdPx_fd(u, I))
print('using dCdP function:', spline.dCdPx(u, I), '\n')

print("Derivative wrt control point Py_{I}: dC/dPy_{I}")
print('using finite differences:', spline.dCdPy_fd(u, I))
print('using dCdP function:', spline.dCdPy(u, I), '\n')

print("Derivative wrt control point Pz_{I}: dC/dPz_{I}")
print('using finite differences:', spline.dCdPz_fd(u, I))
print('using dCdP function:', spline.dCdPz(u, I), '\n')

# NURBS curve (example from pg. 126 of The NURBS Book)
p = 2
P = [[1, 0, 0], [1, 1, 0], [0, 1, 0]]
U = [0, 0, 0, 1, 1, 1]
G = [1, 1, 2]
nurbs = NURBS(P=P, G=G, p=p, U=U)
u = 0
n = 2
I = 0

print("----- NURBS derivatives -----")
print('Curve evaluated at u: C(u)')
print('answer from P&T:', np.array([1.0, 0.0, 0.0]))
print('using point evaluation:', nurbs(u))
print('using deriv function:', nurbs.ders(u, n)[0], '\n')

print("First derivative of curve at u: C'(u)")
print('answer from P&T:', np.array([0.0, 2.0, 0.0]))
print('using finite differences:', nurbs.dCdu_fd(u))
print('using ders function:', nurbs.ders(u, n)[1], '\n')

print("Second derivative of curve at u: C''(u)")
print('answer from P&T:', np.array([-4.0, 0.0, 0.0]))
print('using finite differences:', nurbs.d2Cdu2_fd(u))
print('using ders function:', nurbs.ders(u, n)[2], '\n')

print(nurbs.ders(u, n))
print(f'3rd der =  {nurbs.ders(u, 4)}')


print("Derivative wrt control point Px_{I}: dC/dPx_{I}")
print('using finite differences:', nurbs.dCdPx_fd(u, I))
print('using dCdP function:', nurbs.dCdPx(u, I), '\n')

print("Derivative wrt control point Py_{I}: dC/dPy_{I}")
print('using finite differences:', nurbs.dCdPy_fd(u, I))
print('using dCdP function:', nurbs.dCdPy(u, I), '\n')

print("Derivative wrt control point Pz_{I}: dC/dPz_{I}")
print('using finite differences:', nurbs.dCdPz_fd(u, I))
print('using dCdP function:', nurbs.dCdPz(u, I), '\n')
