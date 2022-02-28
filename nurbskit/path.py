"""
Spline-based path classes.

Author: Reece Otto 11/09/2020
"""
import numpy as np
from math import comb
from .utils import find_span, basis_funs, weighted_control_points, \
                   ders_basis_funs, one_basis_fun, auto_knot_vector
import copy

class Path():
    # Base class for all path objects.
    def __repr__(self):
        pass
    
    def __call__(self, u):
        pass
    
    def u_start(self):
        # Calculates parametric starting point of path.
        return self.U[self.p]
    
    def u_end(self):
        # Calculates parametric ending point of path.
        return self.U[-(self.p + 1)]
    
    def u_midpoint(self):
        # Calculates the parametric mid-point of path.
        return (self.u_start() + self.u_end()) / 2

    def list_eval(self, u_i=0, u_f=1, n_points=100):
        """
        Evalutes points along a path
        
        Keyword arguments:
            u_i = initial u coordinate
            u_f = final u coordinate
            n_points = number of point evaluated along path
        """
        u_vals = np.linspace(u_i, u_f, n_points)
        coords = np.nan * np.ones((n_points, len(self.P[0])))
        for i in range(n_points):
            coords[i] = self(u_vals[i])
        return coords

    def knot_eval(self):
        # evaluates the Cartesian coordinates of each knot.
        return np.array([self(u) for u in self.U])

    def dCdu_fd(self, u):
        # Calculates dC/du at u using finite differences.
        du = 0.001
        C0 = self(u)
        # backwards difference if near path end
        if u + du > self.u_end():
            der = (C0 - self(u - du)) / du
        # forward difference if near path start
        elif u - du < self.u_start():
            der = (self(u + du) - C0) / du
        # central difference if anywhere else
        else:
            der = (self(u + du) - self(u - du)) / (2.0 * du)
        return der
    
    def d2Cdu2_fd(self, u):
        # Calculates d^2C/du^2 at u using finite differences.
        du = 0.001
        C0 = self(u)
        # backwards difference if near path end
        if u + du > self.u_end():
            der = (C0 - 2 * self(u - du) + self(u - 2 * du)) / (du**2)
        # forward difference if near path start
        elif u - du < self.u_start():
            der = (self(u + 2 * du) - 2 * self(u + du) + C0) / (du**2)
        # central difference if anywhere else
        else:
            der = (self(u + du) - 2*C0 + self(u - du)) / (du**2)
        return der

    def dCdPx_fd(self, u, I):
        # Calculates derivative of path wrt to control point Px_{I} at u.
        h = 0.001
        path_pert = copy.deepcopy(self)
        path_pert.P[I][0] += h
        return (path_pert(u) - self(u)) / h
    
    def dCdPy_fd(self, u, I):
        # Calculates derivative of path wrt to control point Py_{I} at u.
        h = 0.001
        path_pert = copy.deepcopy(self)
        path_pert.P[I][1] += h
        return (path_pert(u) - self(u)) / h
    
    def dCdPz_fd(self, u, I):
        # Calculates derivative of path wrt to control point Pz_{I} at u.
        h = 0.001
        path_pert = copy.deepcopy(self)
        path_pert.P[I][2] += h
        return (path_pert(u) - self(u)) / h
    
class Bezier(Path):
    """
    Bezier curve object.
    
    Attributes:
        P = control points
        p = degree of path
        U = knot vector
    """
    def __init__(self, P):
        # check that at least two control points were supplied
        if len(P) < 2:
            str_1 = 'A Bezier path requires at least two control points. \n'
            str_2 = f'Supplied number of control points = {len(P)}'
            raise AttributeError(str_1 + str_2)

        # check if all control points are either 2D or 3D
        dims = np.nan * np.ones(len(P))
        for i in range(len(P)):
            dims[i] = len(P[i])
        if not np.all(dims == dims[0]):
            str_1 = 'Not all control points have the same dimension.'
            raise AttributeError(str_1)
        if not (np.all(dims == 2) or np.all(dims == 3)):
            raise AttributeError('Control points have invalid dimension.')
        del dims

        # assign control points, degree and knot vector to Bezier path object
        self.P = np.array(P)
        self.p = int(len(self.P) - 1)
        self.U = np.nan * np.ones(len(self.P) + self.p + 1)
        self.U[:self.p+1] = 0.0
        self.U[self.p+1:] = 1.0

    def __repr__(self):
        return f"Bezier(P={self.P}, p={self.p})"
    
    def __call__(self, u):
        # deCasteljau's algorithm
        Q = np.zeros((len(self.P), len(self.P[0])))
        for i in range(self.p+1):
            Q[i] = self.P[i]
        for k in range(1, self.p+1):
            for i in range(self.p-k+1):
                Q[i] = (1.0 - u) * Q[i] + u * Q[i+1]
        return Q[0]
    
class BSpline(Path):
    """
    B-Spline curve object.
    
    Attributes:
        P = control points
        p = degree of curve
        U = knot vector
    """
    def __init__(self, P, p, U):
        # check that at least two control points were supplied
        if len(P) < 2:
            str_1 = 'A B-Spline path requires at least two control points. \n'
            str_2 = f'Supplied number of control points = {len(P)}'
            raise AttributeError(str_1 + str_2)

        # check if all control points are either 2D or 3D
        dims = np.nan * np.ones(len(P))
        for i in range(len(P)):
            dims[i] = len(P[i])
        if not np.all(dims == dims[0]):
            str_1 = 'Not all control points have the same dimension.'
            raise AttributeError(str_1)
        if not (np.all(dims == 2) or np.all(dims == 3)):
            raise AttributeError('Control points have invalid dimension.')
        del dims

        # check if path degree is valid
        if p > len(P) - 1:
            str_1 = 'Degree of B-Spline path is not compatible with' + \
            'given number of control points. \n'
            str_2 = f'Supplied path degree: {p}. \n'
            str_3 = f'Maximum allowed path degree: {len(P)-1}.'
            raise AttributeError(str_1 + str_2 + str_3)

        # check if knot vector is valid
        if len(U) != len(P) + p + 1:
            str_1 = 'Knot vector is not compatible with given number of ' + \
                'control points and path degree. \n'
            str_2 = f'Supplied knot vector length: {len(U)}. \n'
            str_3 = f'Required knot vector length: {len(P) + p + 1}.'
            raise AttributeError(str_1 + str_2 + str_3)

        # assign attributes of B-Spline path object
        self.P = np.array(P)
        self.p = p
        self.U = np.array(U)

    
    def __repr__(self):
        return f"BSpline(P={self.P}, p={self.p}, U={self.U})"
    
    def __call__(self, u):
        # de Boor's algorithm
        span = find_span(self.p, u, self.U)
        B = basis_funs(span, u, self.p, self.U)
        C = 0
        for i in range(self.p + 1):
            C += B[i] * np.array(self.P[span-self.p+i])
        return C
    
    def ders(self, u, d):
        """
        Calculates all derivatives wrt u up to order d.
        This is A3.2 on pg 93 of
        'The NURBS Book' - Les Piegl & Wayne Tiller, 1997

        Arguments:
            u = parametric coordinate
            d = order of last derivative
        """
        CK = np.nan * np.ones(shape=(d+1, len(self.P[0])))
        du = min(d, self.p)
        for k in range(self.p+1, d+1):
            CK[k] = 0.0
        span = find_span(self.p, u, self.U)
        nders = ders_basis_funs(span, u, self.p, du, self.U)
        #print(nders)
        for k in range(du+1):
            CK[k] = 0.0
            for j in range(self.p+1):
                CK[k] += nders[k][j] * np.array(self.P[span-self.p+j])
        return CK
    
    def dCdPx(self, u, I):
        # Calculates derivative of path wrt to control point Px_{I} at u.
        NIp = one_basis_fun(I, u, self.p, self.U)
        span = find_span(self.p, u, self.U)
        B = basis_funs(span, u, self.p, self.U)
        denom = 0.0
        for i in range(self.p + 1):
            denom += B[i]
        unit_vec = np.zeros_like(self(u))
        unit_vec[0] = 1
        return NIp * unit_vec / denom
    
    def dCdPy(self, u, I):
        # Calculates derivative of path wrt to control point Py_{I} at u.
        NIp = one_basis_fun(I, u, self.p, self.U)
        span = find_span(self.p, u, self.U)
        B = basis_funs(span, u, self.p, self.U)
        denom = 0.0
        for i in range(self.p + 1):
            denom += B[i]
        unit_vec = np.zeros_like(self(u))
        unit_vec[1] = 1
        return NIp * unit_vec / denom
    
    def dCdPz(self, u, I):
        # Calculates derivative of path wrt to control point Pz_{I} at u.
        NIp = one_basis_fun(I, u, self.p, self.U)
        span = find_span(self.p, u, self.U)
        B = basis_funs(span, u, self.p, self.U)
        denom = 0.0
        for i in range(self.p + 1):
            denom += B[i]
        unit_vec = np.zeros_like(self(u))
        unit_vec[2] = 1
        return NIp * unit_vec / denom
    
class NURBS(Path):
    """
    NURBS curve object.
    
    Attributes:
        P = control points
        G = control point weights
        p = degree of curve
        U = knot vector
    """
    def __init__(self, P, G, p, U):
        # check that at least two control points were supplied
        if len(P) < 2:
            str_1 = 'A NURBS path requires at least two control points. \n'
            str_2 = f'Supplied number of control points = {len(P)}'
            raise AttributeError(str_1 + str_2)

        # check if all control points are either 2D or 3D
        dims = np.nan * np.ones(len(P))
        for i in range(len(P)):
            dims[i] = len(P[i])
        if not np.all(dims == dims[0]):
            str_1 = 'Not all control points have the same dimension.'
            raise AttributeError(str_1)
        if not (np.all(dims == 2) or np.all(dims == 3)):
            raise AttributeError('Control points have invalid dimension.')
        del dims

        # check if there is a weight for each control point
        if len(G) != len(P):
            str_1 = 'The number of control points and the number ' + \
                'of control point weights are not equal. \n'
            str_2 = f'Number of control points: {len(P)}. \n'
            str_3 = f'Number of control point weights: {len(G)}.'
            raise AttributeError(str_1 + str_2 + str_3)

        # check if path degree is valid
        if p > len(P) - 1:
            str_1 = 'Degree of NURBS path is not compatible with' + \
            'given number of control points. \n'
            str_2 = f'Supplied path degree: {p}. \n'
            str_3 = f'Maximum allowed path degree: {len(P)-1}.'
            raise AttributeError(str_1 + str_2 + str_3)

        # check if knot vector is valid
        if len(U) != len(P) + p + 1:
            str_1 = 'Knot vector is not compatible with given number of ' + \
                'control points and path degree. \n'
            str_2 = f'Supplied knot vector length: {len(U)}. \n'
            str_3 = f'Required knot vector length: {len(P) + p + 1}.'
            raise AttributeError(str_1 + str_2 + str_3)

        # assign attributes of B-Spline path object
        self.P = np.array(P)
        self.G = np.array(G)
        self.p = p
        self.U = np.array(U)
        
    def __repr__(self):
        return f"NURBS(P={self.P}, G={self.G}, p={self.p}, U={self.U})"
    
    def __call__(self, u):
        # A4.1 on pg 124 of 'The NURBS Book' - Les Piegl & Wayne Tiller, 1997
        Pw = weighted_control_points(self.P, self.G)
        span = find_span(self.p, u, self.U)
        B = basis_funs(span, u, self.p, self.U)
        Cw = 0
        for j in range(self.p + 1):
            Cw += B[j] * Pw[span-self.p+j]
        C = np.zeros(len(Cw) - 1)
        for k in range(len(C)):
            C[k] = Cw[k] / Cw[-1]
        return C
    
    def ders(self, u, d):
        """
        TODO: this function does not work

        CK = np.nan * np.ones(shape=(d+1, len(self.P[0])))
        du = min(d, self.p)
        for k in range(self.p+1, d+1):
            CK[k] = 0.0
        span = find_span(self.p, u, self.U)
        nders = ders_basis_funs(span, u, self.p, du, self.U)
        for k in range(du+1):
            CK[k] = 0.0
            for j in range(self.p+1):
                CK[k] += nders[k][j] * np.array(self.P[span-self.p+j])
        return CK
        """

        # calculate Cw(u) derivatives using modified version of
        # A3.2 on pg 93 of 'The NURBS Book' - Les Piegl & Wayne Tiller, 1997
        Pw = weighted_control_points(self.P, self.G)
        CwK = np.nan * np.ones(shape=(d+1, len(Pw[0])))
        du = min(d, self.p)
        for k in range(self.p+1, d+1):
            CwK[k] = 0.0
        span = find_span(self.p, u, self.U)
        nders = ders_basis_funs(span, u, self.p, du, self.U)
        print(nders)
        for k in range(du+1):
            CwK[k] = 0.0
            for j in range(self.p+1):
                CwK[k] += nders[k][j] * np.array(Pw[span-self.p+j])

        # calculate C(u) derivatives from Cw(u) derivatives
        # A4.2 on pg 127 of 'The NURBS Book' - Les Piegl & Wayne Tiller, 1997
        Aders = CwK[:,:-1]
        wders = CwK[:,-1]
        CK = np.nan * np.ones(shape=(d+1, len(self.P[0])))
        for k in range(d+1):
            v = Aders[k]
            for i in range(1, k+1):
                """
                print('k =', k)
                print('i =', i)
                print('comb =', comb(k, i))
                print('wders =', wders[i])
                print('k-i =', k-i)
                print('CK =', CK[k-i], '\n')
                """
                v += -comb(k, i) * wders[i] * CK[k-i]
            CK[k] = v / wders[0]
        return CK
    
    def dCdPx(self, u, I):
        # Calculates derivative of path wrt to control point Px_{I} at u.
        NIp = one_basis_fun(I, u, self.p, self.U)
        span = find_span(self.p, u, self.U)
        B = basis_funs(span, u, self.p, self.U)
        denom = 0.0
        for i in range(self.p + 1):
            denom += B[i] * self.G[span-self.p+i]
        unit_vec = np.zeros_like(self(u))
        unit_vec[0] = 1
        return NIp * unit_vec / denom
        return NIp * self.G[I] * unit_vec / denom
    
    def dCdPy(self, u, I):
        # Calculates derivative of path wrt to control point Py_{I} at u.
        NIp = one_basis_fun(I, u, self.p, self.U)
        span = find_span(self.p, u, self.U)
        B = basis_funs(span, u, self.p, self.U)
        denom = 0.0
        for i in range(self.p + 1):
            denom += B[i] * self.G[span-self.p+i]
        unit_vec = np.zeros_like(self(u))
        unit_vec[1] = 1
        return NIp * unit_vec / denom
        return NIp * self.G[I] * unit_vec / denom
    
    def dCdPz(self, u, I):
        # Calculates derivative of path wrt to control point Pz_{I} at u.
        NIp = one_basis_fun(I, u, self.p, self.U)
        span = find_span(self.p, u, self.U)
        B = basis_funs(span, u, self.p, self.U)
        denom = 0.0
        for i in range(self.p + 1):
            denom += B[i] * self.G[span-self.p+i]
        unit_vec = np.zeros_like(self(u))
        unit_vec[2] = 1
        return NIp * unit_vec / denom
        return NIp * self.G[I] * unit_vec / denom
    
    def dCdG(self, u, I):
        # Calculates the derivative of the path wrt to control weight G_{I}.
        NIp = one_basis_fun(I, u, self.p, self.U)
        span = find_span(self.p, u, self.U)
        B = basis_funs(span, u, self.p, self.U)
        denom = 0.0
        for i in range(self.p + 1):
            denom += B[i] * self.G[span-self.p+i]
        return NIp * (self.P[i] - self(u))/ denom
    
    def dCdG_fd(self, u, I):
        # Calculates derivative of path wrt to control weight G_{I} using 
        # finite differences.
        h = 0.001
        path_pert = copy.deepcopy(self)
        path_pert.G[I] += h
        return (path_pert(u) - self(u)) / h

def Ellipse(a, b, h=0, k=0):
    p = 2
    P = [[-a + h, k], [-a + h, b + k], [h, b + k], [a + h, b + k], [a + h, k],
         [a + h, -b + k], [h, -b + k], [-a + h, -b + k], [-a + h, k]]
    G = [1, 1, 2, 1, 1, 1, 2, 1, 1]
    U = [0, 0, 0, 1/4, 1/4, 1/2, 1/2, 3/4, 3/4, 1, 1, 1]
    return NURBS(P=P, p=p, G=G, U=U)

def TopEllipse(a, b, h=0, k=0):
    p = 2
    P = [[a + h, k], [a + h, b + k], [h, b + k], [-a + h, b + k], [-a + h, k]]
    G = [1, 1, 2, 1, 1]
    U = [0, 0, 0, 1/2, 1/2, 1, 1, 1]
    return NURBS(P=P, p=p, G=G, U=U)

def Rectangle(width, height, centre_x=0, centre_y=0):
    P = [[centre_x-width/2, centre_y], [centre_x-width/2, centre_y+height/2],
     [centre_x-width/2, centre_y+height/2], [centre_x, centre_y+height/2],
     [centre_x+width/2, centre_y+height/2], 
     [centre_x+width/2, centre_y+height/2], [centre_x+width/2, centre_y],
     [centre_x+width/2, centre_y-height/2], 
     [centre_x+width/2, centre_y-height/2],
     [centre_x, centre_y-height/2], [centre_x-width/2, centre_y-height/2], 
     [centre_x-width/2, centre_y-height/2], [centre_x-width/2, centre_y]]
    p = 2
    U = auto_knot_vector(len(P), p)
    return BSpline(P=P, p=p, U=U)