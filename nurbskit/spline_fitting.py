import numpy as np
from math import ceil
from .utils import basis_funs, find_span, auto_knot_vector
from nurbskit.point_inversion import point_proj_path
from nurbskit.path import BSpline
from scipy.optimize import minimize

# remove these
import matplotlib.pyplot as plt

def global_curve_interp(Q, p):
    """
    Global curve interpolation through n+1 points.
    This is algorithm A9.1 on pg 369 of:
    'The NURBS Book' - Les Piegl & Wayne Tiller, 1997.
    
    Arguments:
        Q = list of points to be interpolated
        p = degree of interpolative B-Spline curve
    """
    n = int(len(Q) - 1)
    m = n + p + 1

    d = 0.0
    for k in range(1, n+1):
        d += np.linalg.norm(np.array(Q[k]) - np.array(Q[k-1]))

    # compute the u_k
    U_bar = [None] * (n+1)
    U_bar[0] = 0.0
    U_bar[-1] = 1.0
    for k in range(1, n):
        U_bar[k] = U_bar[k-1] + np.linalg.norm(np.array(Q[k]) - \
            np.array(Q[k-1])) / d

    # compute the knot vector U
    U = [None] * (m + 1)
    for i in range(p+1):
        U[i] = 0.0
        U[m-p+i] = 1.0

    for j in range(1, n-p+1):
        s = 0.0
        for i in range(j, j+p):
            s += U_bar[i]
        U[j+p] = s / p

    # fill coefficient matrix
    A = np.zeros((n+1, n+1))
    for i in range(n+1):
        span = find_span(p, U_bar[i], U)
        A[i][span-p:span+1] = basis_funs(span, U_bar[i], p, U)

    # compute control points
    P = np.linalg.solve(A, Q)

    return U, P

def fit_bspline_path(Q, N_P, p, tol=1E-6):
    """
    Use optimization to fit a B-Spline path to given point data.
    Note: this function is only suited for non-complex geometries

    Arguments:
        Q = point data
        N_P = number of control points
        p = degree of B-Spline path

    Keyword arguments:
        tol = fitting tolerance
    """
    # ensure that there are more data points than control points
    if len(Q) < N_P:
        raise ValueError('Number of control points must be larger than ' + \
            'number of data points.')

    # ensure that path degree is compatible with N_P
    if p > N_P - 1:
        raise ValueError('Path degree is not compatible with number of ' + \
            'control points.')

    # take subset of point data as initial guess for control point positions
    delta_ind = (len(Q) - 1) / (N_P - 1)
    P_guess = np.nan * np.ones((N_P, len(Q[0])))
    for i in range(N_P):
        ind = int(ceil(i * delta_ind))
        P_guess[i] = Q[ind]
    
    # create initial B-Spline path
    U = auto_knot_vector(N_P, p)
    spline = BSpline(P=P_guess, p=p, U=U)

    # create objective function based on the sum of distances between data
    # point and closest point on path
    def sum_dist(design_vars, spline, Q):
        # create new B-Spline based on perturbed design_vars
        P_new = np.reshape(design_vars, spline.P.shape)
        spline_new = BSpline(P=P_new, p=p, U=U)
        
        # find closest point on curve for each data point
        f = 0
        for i in range(len(Q)):
            u_proj = point_proj_path(spline_new, Q[i], tol_dist=1E-3, 
                tol_cos=1E-3, tol_end=1E-3)
            f += np.linalg.norm(spline_new(u_proj) - Q[i])**2
        return f

    def call_back(design_vars):
        f = sum_dist(design_vars, spline, Q)
        global it_no
        print(f'Iteration={it_no} f={f:.4}')
        it_no += 1

    # minimize objective function
    design_vars = P_guess.flatten()
    global it_no
    it_no = 0
    print('Fitting B-Spline path to point data. \n')
    sol = minimize(sum_dist, design_vars, method='Nelder-Mead', 
        callback=call_back, args=(spline, Q), options={'disp':True, 
        'maxiter':200})

    # check if optimizer converged
    if sol.success == False:
        raise AssertionError('Optimizer did not converge.')

    # return B-Spline path
    P_sol = sol.x.reshape(spline.P.shape)
    return BSpline(P=P_sol, p=p, U=U)

def surf_mesh_params(Q):
    """
    Computes surface mesh parameters for global surface interpolation.

    This is algorithm A9.3 on pg 377 of:
    'The NURBS Book' - Les Piegl & Wayne Tiller, 1997.

    Arguments:
        Q = list of points to be interpolated 
    """
    n = int(len(Q) - 1)
    m = int(len(Q[0]) - 1)

    # compute uk
    num = m + 1

    uk = [None] * (n+1)
    uk[0] = 0.0
    uk[n] = 0.0
    cds = [None] * (n+1)
    for k in range(1, n):
        uk[k] = 0.0
    for l in range(m+1):
        total = 0.0
        for k in range(1, n+1):
            cds[k] = np.linalg.norm(np.array(Q[k][l]) - np.array(Q[k-1][l]))
            total += cds[k]
        if total == 0.0:
            num = num - 1
        else:
            d = 0.0;
            for k in range(1, n):
                d += cds[k]
                uk[k] += d / total

    if num == 0.0:
        raise ValueError("num == 0.0")
    for k in range(1, n):
        uk[k] = uk[k] / num

    # compute vl
    num = n + 1

    vl = [None] * (m+1)
    vl[0] = 0.0
    vl[m] = 0.0
    cds = [None] * (m+1)
    for l in range(1, m):
        vl[l] = 0.0
    for k in range(n+1):
        total = 0.0
        for l in range(1, m+1):
            cds[l] = np.linalg.norm(np.array(Q[k][l]) - np.array(Q[k][l-1]))
            total += cds[l]
        if total == 0.0:
            num = num - 1
        else:
            d = 0.0;
            for l in range(1, m):
                d += cds[l]
                vl[l] += d / total

    if num == 0.0:
        raise ValueError("num == 0.0")
    for l in range(1, m):
        vl[l] = vl[l] / num

    return uk, vl

def global_surf_interp(Q, p, q):
    """
    Global surface interpolation through (n+1) x (m+1) points.
    This is algorithm A9.4 on pg 380 of:
    'The NURBS Book' - Les Piegl & Wayne Tiller, 1997.
    
    Arguments:
        Q = list of points to be interpolated
        p = degree of interpolative B-Spline surface in u direction
        q = degree of interpolative B-Spline surface in v direction
    """
    n = int(len(Q) - 1)
    m = int(len(Q[0]) - 1)

    uk, vl = surf_mesh_params(Q)

    # compute the knot vector U
    m_u = n + p + 1
    U = [None] * (m_u + 1)
    for i in range(p+1):
        U[i] = 0.0
        U[m_u-p+i] = 1.0

    for j in range(1, n-p+1):
        s = 0.0
        for i in range(j, j+p):
            s += uk[i]
        U[j+p] = s / p

    # compute the knot vector V
    m_v = m + q + 1
    V = [None] * (m_v + 1)
    for i in range(q+1):
        V[i] = 0.0
        V[m_v-q+i] = 1.0

    for j in range(1, m-q+1):
        s = 0.0
        for i in range(j, j+q):
            s += vl[i]
        V[j+q] = s / q

    # curve interpolation through Q[0][l], ..., Q[n][l]
    # this yields R[0][l], ..., R[n][l]
    R = np.zeros((n+1, m+1, len(Q[0][0])))
    for l in range(m+1):
        R[:, l] = global_curve_interp(np.array(Q)[:, l], p)[1]
    
    # curve interpolation through R[i][0], ..., R[i][m]
    # this yields P[i][0], ..., P[i][m]
    P = np.zeros((n+1, m+1, len(Q[0][0])))
    for i in range(n+1):
        P[i, :] = global_curve_interp(np.array(Q)[i, :], q)[1]
    
    return U, V, P
