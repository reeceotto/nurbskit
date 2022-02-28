"""
Utility functions for geometric objects.

Author: Reece Otto 11/09/2020
"""
import numpy as np
import scipy.linalg as scilinalg
from math import sqrt

def find_span(p, u, U):
    """
    Returns the index of the knot vector whose value is less than the parameter.
    This is algorithm A2.1 on pg 68 of:
    'The NURBS Book' - Les Piegl & Wayne Tiller, 1997.
    
    Arguments:
        p = degree of curve
        u = parameteric coordinate
        U = knot vector
    """
    if u < U[0] or u > U[-1]:
        raise IndexError(f'u == {u} out of range: [{U[0]}, {U[-1]}]')
    m = len(U) - 1
    n = m - p - 1
    if u == U[n+1]: return n # Special case
    # Do binary search
    low = p
    high = n + 1
    mid = (low + high) // 2
    while u < U[mid] or u >= U[mid+1]:
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid

def basis_funs(i, u, p, U):
    """
    Returns list of all non-zero B-Spline basis functions.
    This is algorthim A2.2 on pg 70 of:
    'The NURBS Book' - Les Piegl & Wayne Tiller, 1997.
    
    Arguments:
        i = index
        u = parameteric coordinate
        p = degree of curve
        U = knot vector
    """
    B = np.nan * np.ones(p+1)
    B[0] = 1.0
    left = np.nan * np.ones_like(B)
    right = np.nan * np.ones_like(B)
    for j in range(1, p+1):
        left[j] = u - U[i+1-j]
        right[j] = U[i+j] - u
        saved = 0.0
        for r in range(j):
            temp = B[r] / (right[r+1] + left[j-r])
            B[r] = saved + right[r+1] * temp
            saved = left[j-r] * temp
        B[j] = saved
    return B

def one_basis_fun(i, u, p, U):
    """
    Returns a single basis function Ni,p(u).

    This is algorthim A2.4 on pg 74 of:
    'The NURBS Book' - Les Piegl & Wayne Tiller, 1997.
    
    Arguments:
        i = index
        u = parameteric coordinate
        p = degree of curve
        U = knot vector
    """
    m = len(U) - 1
    if (i == 0 and u == U[0]) or (i == m - p - 1 and u == U[m]):
        return 1.0
    if (u < U[i] or u >= U[i + p + 1]):
        return 0.0
    N = np.nan * np.ones(p+1)
    for j in range(p+1):
        if u >= U[i+j] and u < U[i+j+1]:
            N[j] = 1.0
        else:
            N[j] = 0.0
    for k in range(1, p+1):
        if N[0] == 0:
            saved = 0.0
        else:
            saved = ((u - U[i]) * N[0]) / (U[i+k] - U[i])
        for j in range(p-k+1):
            Uleft = U[i+j+1]
            Uright = U[i+j+k+1]
            if N[j+1] == 0:
                N[j] = saved
                saved = 0
            else:
                temp = N[j+1] / (Uright - Uleft)
                N[j] = saved + (Uright - u) * temp
                saved = (u - Uleft) * temp
    Nip = N[0]
    return Nip

def ders_basis_funs(i, u, p, n, U):
    """
    Compute non-zero basis functions and their derivatives up to degree n.

    This is algorithm A2.3 on pg 72 of:
    'The NURBS Book' - Les Piegl & Wayne Tiller, 1997.

    Arguments:
        i = index
        u = parameteric coordinate
        p = degree of curve
        n = order of last derivative
        U = knot vector
    """
    ndu = np.nan * np.ones(shape=(p+1, p+1))
    a = np.nan * np.ones(shape=(2, p+1))
    ders = np.nan * np.ones(shape=(n+1, p+1))
    left = np.nan * np.ones(p+1)
    right = np.nan * np.ones(p+1)
    ndu[0][0] = 1.0
    for j in range(1, p+1):
        left[j] = u - U[i+1-j]
        right[j] = U[i+j] - u
        saved = 0.0
        for r in range(j):
            ndu[j][r] = right[r+1] + left[j-r]
            temp = ndu[r][j-1] / ndu[j][r]
            ndu[r][j] = saved + right[r+1] * temp
            saved = left[j-r] * temp
        ndu[j][j] = saved
    for j in range(p+1):
        ders[0][j] = ndu[j][p]
    # TODO: error somewhere past here
    for r in range(p+1):
        s1 = 0
        s2 = 1
        a[0][0] = 1.0
        for k in range(1, n+1):
            d = 0.0
            rk = r - k
            pk = p - k
            if r >= k:
                a[s2][0] = a[s1][0] / ndu[pk+1][rk]
                d = a[s2][0] * ndu[rk][pk]
            if rk >= -1:
                j1 = 1
            else:
                j1 = -rk

            if r - 1 <= pk:
                j2 = k - 1
            else:
                j2 = p - r

            for j in range(j1, j2+1):
                a[s2][j] = (a[s1][j] - a[s1][j-1]) / ndu[pk+1][rk+j]
                d += a[s2][j] * ndu[rk+j][pk]
            if r <= pk:
                a[s2][k] = -a[s1][k-1] / ndu[pk+1][r]
                d += a[s2][k] * ndu[r][pk]
            ders[k][r] = d
            j = s1
            s1 = s2
            s2 = j
    r = p
    for k in range(1, n+1):
        for j in range(p+1):
            ders[k][j] *= r
        r *= (p - k)

    return ders

def ders_one_basis_fun(i, u, p, n, U):
    """
    Computes derivatives of basis function Nip.

    This is algorithm A2.5 on pg 76 of:
    'The NURBS Book' - Les Piegl & Wayne Tiller, 1997.

    Arguments:
        i = index
        u = parameteric coordinate
        p = degree of curve
        n = order of last derivative
        U = knot vector
    """
    ders = np.nan * np.ones(n+1)
    N = np.nan * np.ones(shape=(n+1, p+1))
    ND = np.nan * np.ones(n+1)
    if u < U[i] or u >= U[i+p+1]:
        for k in range(n+1):
            ders[k] = 0.0
            return ders
    for j in range(p+1):
        if u >= U[i+j] and u < U[i+j+1]:
            N[j][0] = 1.0
        else:
            N[j][0] = 0.0
    for k in range(1, p+1):
        if N[0][k-1] == 0.0:
            saved = 0.0
        else:
            saved = ((u - U[i]) * N[0][k-1]) / (U[i+k] - U[i])
        for j in range(p-k+1):
            Uleft = U[i+j+1]
            Uright = U[i+j+k+1]
            if N[j+1][k-1] == 0.0:
                N[j][k] = saved
                saved = 0.0
            else:
                temp = N[j+1][k-1] / (Uright - Uleft)
                N[j][k] = saved + (Uright - u) * temp
                saved = (u - Uleft) * temp
    ders[0] = N[0][p]
    for k in range(1, n+1):
        for j in range(k+1):
            ND[j] = N[j][p-k]
        for jj in range(1, k+1):
            if ND[0] == 0.0:
                saved = 0.0
            else:
                saved = ND[0] / (U[i+p-k+jj] - U[i])
            for j in range(k-jj+1):
                Uleft = U[i+j+1]
                Uright = U[i+j+p+jj+1]
                if ND[j+1] == 0.0:
                    ND[j] = (p - k + jj) * saved
                    saved = 0.0
                else:
                    temp = ND[j+1] / (Uright - Uleft)
                    ND[j] = (p - k + jj) * (saved - temp)
                    saved = temp
        ders[k] = ND[0]
    return ders

def auto_knot_vector(n, p):
    """
    Constructs a knot vector that is clamped, uniform and normalised.

    This is Eqn 5 from 'Freeform Deformation Versus B-Spline Representation in
    Inverse Airfoil Design'
    - Eleftherios I. Amoiralis, Ioannis K. Nikolos, 2008
    
    Arguments:
        n = number of control points
        p = degree of curve
    """
    a = n - 1
    if p > a:
        if p < 1:
            raise ValueError("1 <= p <= a condition not satisfied!")
    q = a + p + 1
    U = [None] * (q + 1)
    for i in range(len(U)):
        if 0 <= i:
            if i <= p:
                U[i] = 0
        if p < i:
            if i <= q - p - 1:
                U[i] = i - p
        if q - p - 1 < i:
            if i <= q:
                U[i] = q - 2 * p
    for i in range(len(U)):
        U[i] = U[i] / max(U)
    return U

def weighted_control_points(P, G):
    """
    Returns weighted control point tensor.
    
    Arguments:
        P = array of control points
        G = array of control point weights
    """

    # ensure that there is one weight for every control point
    #print(int(P.shape[1:]))


    if len(P.shape) == 2:
        Pw = np.zeros((len(P), len(P[0]) + 1))
        for i in range(len(P)):
            Pw[i][len(P[0])] = G[i]
            for j in range(len(P[0])):
                Pw[i][j] = G[i] * P[i][j]

    if len(P.shape) == 3:
        Pw = np.zeros((len(P), len(P[0]), len(P[0][0]) + 1))
        for i in range(len(P)):
            for j in range(len(P[0])):
                Pw[i][j][-1] = G[i][j]
                for k in range(len(P[0][0])):
                    Pw[i][j][k] = G[i][j] * P[i][j][k]

    if len(P.shape) == 4:
        Pw = np.zeros((len(P), len(P[0]), len(P[0][0]), len(P[0][0][0]) + 1))
        for i in range(len(P)):
            for j in range(len(P[0])):
                for k in range(len(P[0][0])):
                    Pw[i][j][k][-1] = G[i][j][k]
                    for l in range(len(P[0][0][0])):
                       Pw[i][j][k][l] = G[i][j][k] * P[i][j][k][l]

    return Pw