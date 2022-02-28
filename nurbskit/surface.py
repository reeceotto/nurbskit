"""
Spline-based surface classes.

Author: Reece Otto 06/10/2021
"""
import numpy as np
from .utils import find_span, basis_funs, weighted_control_points, one_basis_fun

class Surface():
    # Base class for all surface objects.
    def __repr__(self):
        pass
    
    def __call__(self, u, v):
        pass
    
    def u_start(self):
        # Calculates parametric starting point of surface in u direction.
        return self.U[self.p]
    
    def u_end(self):
        # Calculates parametric ending point of surface in u direction.
        return self.U[-(self.p + 1)]
    
    def v_start(self):
        # Calculates parametric starting point of surface in v direction.
        return self.V[self.q]
    
    def v_end(self):
        # Calculates parametric ending point of surface in v direction.
        return self.V[-(self.q + 1)]
    
    def knot_eval(self):
        # returns an array with Cartesian coordinates of knots.
        knots = np.nan * np.ones((len(self.U),len(self.V), len(self.P[0][0])))
        for i in range(len(self.U)):
            for j in range(len(self.V)):
                knots[i][j] = self(self.U[i], self.V[j])
        return knots
    
    def u_closed(self):
        # returns 1 if the surface is closed in the u direction.
        flag = 1
        for j in range(len(self.P[0])):
            if not np.array_equal(self.P[0][j], self.P[int(len(self.P)-1)][j]):
                flag = 0
                break
        return flag
    
    def v_closed(self):
        # returns 1 if the surface is closed in the v direction.
        flag = 1
        for i in range(len(self.P)):
            if not np.array_equal(self.P[i][0], 
                self.P[i][int(len(self.P[0])-1)]):
                flag = 0
                break
        return flag
    
    def u_periodic(self):
        # returns 1 if the surface is NOT clamped (periodic) in u direction.
        u_initial = self.U[0:self.p+1]
        initial_flag =  int(all(elem == u_initial[0] for elem in u_initial))
        u_final = self.U[-1-self.p:]
        final_flag = int(all(elem == u_final[0] for elem in u_final))
        if initial_flag == 1:
            if final_flag == 1:
                return 0
            else:
                return 1
        else:
            return 1
    
    def v_periodic(self):
        # returns 1 if the surface is NOT clamped (periodic) in v direction.
        v_initial = self.V[0:self.q+1]
        initial_flag =  int(all(elem == v_initial[0] for elem in v_initial))
        v_final = self.V[-1-self.q:]
        final_flag = int(all(elem == v_final[0] for elem in v_final))
        if initial_flag == 1:
            if final_flag == 1:
                return 0
            else:
                return 1
        else:
            return 1

    def list_eval(self, u_i=0, u_f=1, N_u=100, v_i=0, v_f=1, N_v=100):
        """
        Evalutes a list of points along a surface.
        
        Keyword arguments:
            u_i = initial u coordinate
            u_f = final u coordinate
            N_u = number of point evaluated along u direction
            v_i = initial v coordinate
            v_f = final v coordinate
            N_v = number of point evaluated along v direction
        """
        u_vals = np.linspace(u_i, u_f, N_u)
        v_vals = np.linspace(v_i, v_f, N_v)
        coords = np.nan * np.ones((N_u, N_v, len(self.P[0][0])))
        for i in range(N_u):
            for j in range(N_v):
                coords[i][j] = self(u_vals[i], v_vals[j])
        return coords

    # surface derivatives
    def dSdu(self, u, v):
        # Calculates dS/du using finite differences.
        du = 0.001
        S0 = self(u, v)
        # backwards difference if near u end
        if u + du > self.U[-(self.p + 1)]:
            der = (S0 - self(u - du, v)) / du
        # forward difference if near u start
        elif u - du < self.U[self.p]:
            der = (self(u + du, v) - S0) / du
        # central difference if anywhere else
        else:
            der = (self(u + du, v) - self(u - du, v)) / (2.0 * du)
        return der
    
    def dSdv(self, u, v):
        # Calculates dS/dv using finite differences.
        dv = 0.001
        S0 = self(u, v)
        # backwards difference if near v end
        if v + dv > self.V[-(self.q + 1)]:
            der = (S0 - self(u, v - dv)) / dv
        # forward difference if near v start
        elif v - dv < self.V[self.q]:
            der = (self(u, v + dv) - S0) / dv
        # central difference if anywhere else
        else:
            der = (self(u, v + dv) - self(u, v - dv)) / (2.0 * dv)
        return der
    
    def d2Sdudv(self, u, v):
        # Calculates d^2S/dudv using finite differences.
        du = 0.001
        dv = 0.001
        S0 = self(u, v)
        # forward difference u if near u start
        if u - du < self.U[self.p]:
            # forward difference v if near v start
            if v - dv < self.V[self.q]:
                der = (self(u + du, v + dv) - self(u + du, v) -
                   self(u, v + dv) + S0) / du / dv
            # backward difference v if near v end
            elif v + dv > self.V[-(self.q + 1)]:
                der = (self(u + du, v) + self(u, v - dv) - 
                   self(u + du, v - dv) - S0) / du / dv
            # central difference v elsewhere
            else:
                der = (self(u + du, v + dv) - self(u + du, v - dv)
                       - self(u, v + dv) + self(u, v - dv)) / (2 * du * dv)
        # backward difference u if near u end
        elif u + du > self.U[-(self.p + 1)]:
            # forward difference v if near v start
            if v - dv < self.V[self.q]:
                der = (self(u - du, v) + self(u, v + dv) - 
                   self(u - du, v + dv) - S0) / du / dv
            # backward difference v if near v end
            elif v + dv > self.V[-(self.q + 1)]:
                der = (self(u - du, v - dv) - self(u - du, v) -
                   self(u, v - dv) + S0) / du / dv
            # central difference v elsewhere:
            else:
                der = (self(u - du, v - dv) - self(u - du, v + dv)
                       - self(u, v - dv) + self(u, v + dv)) / (2 * du * dv)
        # central difference u, forward difference v if near v start
        elif v - dv < self.V[self.q]:
            der = (self(u + du, v + dv) - self(u - du, v + dv)
                   - self(u + du, v) + self(u - du, v)) / (2 * du * dv)
        # central difference u, backward difference v if near v end
        elif v + dv > self.V[-(self.q + 1)]:
            der = (self(u - du, v - dv) - self(u + du, v - dv)
                   - self(u - du, v) + self(u + du, v)) / (2 * du * dv)
        # central difference u, central difference v elsewhere
        else:
            der = (self(u + du, v + dv) - self(u + du, v - dv) - 
                   self(u - du, v + dv) + self(u - du, v - dv)) / (4 * du * dv)
        return der
    
    def d2Sdu2(self, u, v):
        # Calculates d^2S/du^2 using finite differences.
        du = 0.001
        S0 = self(u, v)
        # backwards difference if near u end
        if u + du > self.U[-(self.p + 1)]:
            der = (S0 - 2 * self(u - du, v) + self(u - 2 * du, v)) / (du**2)
        # forward difference if near u start
        elif u - du < self.U[self.p]:
            der = (self(u + 2 * du, v) - 2*self(u + du, v) + S0) / (du**2)
        # central difference if anywhere else
        else:
            der = (self(u + du, v) - 2 * S0 + self(u - du, v)) / (du**2)
        return der
    
    def d2Sdv2(self, u, v):
        # Calculates d^2S/dv^2 using finite differences.
        dv = 0.001
        S0 = self(u, v)
        # backwards difference if near v end
        if v + dv > self.V[-(self.q + 1)]:
            der = (S0 - 2 * self(u, v - dv) + self(u, v - 2 * dv)) / (dv**2)
        # forward difference if near v start
        elif v - dv < self.V[self.q]:
            der = (self(u, v + 2 * dv) - 2 * self(u, v + dv) + S0) / (dv**2)
        # central difference if anywhere else
        else:
            der = (self(u, v + dv) - 2 * S0 + self(u, v - dv)) / (dv**2)
        return der

class BSplineSurface(Surface):
    """
    B-Spline surface object.
    
    P = control points
    p, q = orders of curves
    U, V = knot vectors
    """
    def __init__(self, P, p, q, U, V):
        # check that at least two control points were supplied in each direction
        if len(P) < 2 or len(P[0]) < 2:
            str_1 = 'A B-Spline surface requires at least two control ' + \
            'points along each axis. \n'
            str_2 = f'Number of control points along axis 0 = {len(P)}'
            str_3 = f'Number of control points along axis 1 = {len(P[0])}'
            raise AttributeError(str_1 + str_2 + str_3)

        # check if all control points are either 2D or 3D
        dims = np.nan * np.ones((len(P), len(P[0])))
        for i in range(len(P)):
            for j in range(len(P[0])):
                dims[i][j] = len(P[i][j])
        if not np.all(dims == dims[0]):
            str_1 = 'Not all control points have the same dimension.'
            raise AttributeError(str_1)
        if not (np.all(dims == 2) or np.all(dims == 3)):
            raise AttributeError('Control points have invalid dimension.')
        del dims

        # check if surface degree p is valid
        if p > len(P) - 1:
            str_1 = 'Degree of B-Spline surface, p, is not compatible with' + \
            'given number of control points along axis 0. \n'
            str_2 = f'Supplied path degree: {p}. \n'
            str_3 = f'Maximum allowed path degree: {len(P)-1}.'
            raise AttributeError(str_1 + str_2 + str_3)

        # check if surface degree q is valid
        if q > len(P[0]) - 1:
            str_1 = 'Degree of B-Spline surface, q, is not compatible with' + \
            'given number of control points along axis 1. \n'
            str_2 = f'Supplied path degree: {q}. \n'
            str_3 = f'Maximum allowed path degree: {len(P[0])-1}.'
            raise AttributeError(str_1 + str_2 + str_3)

        # check if knot vector U is valid
        if len(U) != len(P) + p + 1:
            str_1 = 'Knot vector, U, is not compatible with given number of' + \
                ' control points along axis 0 and path degree p. \n'
            str_2 = f'Supplied knot vector length: {len(U)}. \n'
            str_3 = f'Required knot vector length: {len(P) + p + 1}.'
            raise AttributeError(str_1 + str_2 + str_3)

        # check if knot vector V is valid
        if len(V) != len(P[0]) + q + 1:
            str_1 = 'Knot vector, V, is not compatible with given number of' + \
                ' control points along axis 1 and path degree q. \n'
            str_2 = f'Supplied knot vector length: {len(V)}. \n'
            str_3 = f'Required knot vector length: {len(P[0]) + q + 1}.'
            raise AttributeError(str_1 + str_2 + str_3)

        # assign attributes of B-Spline surface object
        self.P = np.array(P)
        self.p = p
        self.q = q
        self.U = np.array(U)
        self.V = np.array(V)
        
    def __repr__(self):
        return f"BSplineSurface(P={self.P}, p={self.p}, q={self.q},\
                 U={self.U}, V={self.V})"

    def __call__(self, u, v):
        # A3.5 on pg 103 of 'The NURBS Book' - Les Piegl & Wayne Tiller, 1997.
        u_span = find_span(self.p, u, self.U)
        Bu = basis_funs(u_span, u, self.p, self.U)
        v_span = find_span(self.q, v, self.V)
        Bv = basis_funs(v_span, v, self.q, self.V)
        uind = u_span - self.p
        S = 0.0
        for l in range(self.q+1):
            temp = 0.0
            vind = v_span - self.q + l
            for k in range(self.p+1):
                temp += Bu[k] * np.array(self.P[uind+k][vind])
            S += Bv[l] * temp
        return S
    
    def cast_to_nurbs_surface(self):
        G = np.ones((len(self.P), len(self.P[0])))
        return NURBSSurface(P=self.P, U=self.U, V=self.V, p=self.p, q=self.q, 
                            G=G)
    
class NURBSSurface(Surface):
    """
    NURBS surface object.
    
    P = control points
    G = control point weights
    p, q = orders of curves
    U, V = knot vectors
    """
    def __init__(self, P, G, p, q, U, V):
        # check that at least two control points were supplied in each direction
        if len(P) < 2 or len(P[0]) < 2:
            str_1 = 'A NURBS surface requires at least two control points \
            along each axis. \n'
            str_2 = f'Number of control points along axis 0 = {len(P)}'
            str_3 = f'Number of control points along axis 1 = {len(P[0])}'
            raise AttributeError(str_1 + str_2 + str_3)

        # check if all control points are either 2D or 3D
        dims = np.nan * np.ones((len(P), len(P[0])))
        for i in range(len(P)):
            for j in range(len(P[0])):
                dims[i][j] = len(P[i][j])
        if not np.all(dims == dims[0]):
            str_1 = 'Not all control points have the same dimension.'
            raise AttributeError(str_1)
        if not (np.all(dims == 2) or np.all(dims == 3)):
            raise AttributeError('Control points have invalid dimension.')
        del dims

        # check if there is a weight for each control point along each axis
        if len(G) != len(P) or len(G[0]) != len(P[0]):
            str_1 = 'The number of control points and the number ' + \
                'of control point weights are not equal along each axis. \n'
            str_2 = f'Number of control points along axis 0: {len(P)}. \n'
            str_3 = f'Number of control point weights along axis 0:' + \
            f' {len(G)}. \n'
            str_4 = f'Number of control points along axis 1: {len(P[0])}. \n'
            str_5 = f'Number of control point weights along axis 1:' + \
            f' {len(G[0])}.'
            raise AttributeError(str_1 + str_2 + str_3 + str_4 + str_5)

        # check if surface degree p is valid
        if p > len(P) - 1:
            str_1 = 'Degree of NURBS surface, p, is not compatible with' + \
            'given number of control points along axis 0. \n'
            str_2 = f'Supplied path degree: {p}. \n'
            str_3 = f'Maximum allowed path degree: {len(P)-1}.'
            raise AttributeError(str_1 + str_2 + str_3)

        # check if surface degree q is valid
        if q > len(P[0]) - 1:
            str_1 = 'Degree of NURBS surface, q, is not compatible with' + \
            'given number of control points along axis 1. \n'
            str_2 = f'Supplied path degree: {q}. \n'
            str_3 = f'Maximum allowed path degree: {len(P[0])-1}.'
            raise AttributeError(str_1 + str_2 + str_3)

        # check if knot vector U is valid
        if len(U) != len(P) + p + 1:
            str_1 = 'Knot vector, U, is not compatible with given number of' + \
                ' control points along axis 0 and path degree p. \n'
            str_2 = f'Supplied knot vector length: {len(U)}. \n'
            str_3 = f'Required knot vector length: {len(P) + p + 1}.'
            raise AttributeError(str_1 + str_2 + str_3)

        # check if knot vector V is valid
        if len(V) != len(P[0]) + q + 1:
            str_1 = 'Knot vector, V, is not compatible with given number of' + \
                ' control points along axis 1 and path degree q. \n'
            str_2 = f'Supplied knot vector length: {len(V)}. \n'
            str_3 = f'Required knot vector length: {len(P[0]) + q + 1}.'
            raise AttributeError(str_1 + str_2 + str_3)

        # assign attributes of NURBS surface object
        self.P = np.array(P)
        self.G = np.array(G)
        self.p = p
        self.q = q
        self.U = np.array(U)
        self.V = np.array(V)
        
    def __repr__(self):
        return f"NURBSSurface(P={self.P}, G={self.G}, p={self.p}, q={self.q},\
                 U={self.U}, V={self.V})"
    
    def __call__(self, u, v):
        # A4.3 on pg 134 of 'The NURBS Book' - Les Piegl & Wayne Tiller, 1997.
        Pw = weighted_control_points(self.P, self.G)
        u_span = find_span(self.p, u, self.U)
        Bu = basis_funs(u_span, u, self.p, self.U)
        v_span = find_span(self.q, v, self.V)
        Bv = basis_funs(v_span, v, self.q, self.V)
        temp = np.nan * np.ones(self.q + 1, dtype=np.ndarray)
        for l in range(self.q + 1):
            temp[l] = 0
            for k in range(self.p + 1):
                temp[l] += Bu[k] * np.array(Pw[u_span-self.p+k]
                                              [v_span-self.q+l])
        Sw = 0
        for l in range(self.q + 1):
            Sw += Bv[l] * temp[l]
        S = np.zeros(len(Sw)-1)
        for k in range(len(S)):
            S[k] = Sw[k] / Sw[-1]
        return S
    
    def dSdP(self, u, v, I, J):
        # Calculates dS/dP_{IJ}
        NIp = one_basis_fun(I, u, self.p, self.U)
        NJq = one_basis_fun(J, v, self.q, self.V)
        u_span = find_span(self.p, u, self.U)
        Bu = basis_funs(u_span, u, self.p, self.U)
        v_span = find_span(self.q, v, self.V)
        Bv = basis_funs(v_span, v, self.q, self.V)
        uind = u_span - self.p
        denom = 0.0
        for l in range(self.q+1):
            temp = 0.0
            vind = v_span - self.q + l
            for k in range(self.p+1):
                temp += Bu[k] * self.G[uind+k][vind]
            denom += Bv[l] * temp
        return NIp * NJq * self.G[I][J] / denom
    
    def dSdG(self, u, v, I, J):
        # Calculates dS/dG_{IJ}
        NIp = one_basis_fun(I, u, self.p, self.U)
        NJq = one_basis_fun(J, v, self.q, self.V)
        u_span = find_span(self.p, u, self.U)
        Bu = basis_funs(u_span, u, self.p, self.U)
        v_span = find_span(self.q, v, self.V)
        Bv = basis_funs(v_span, v, self.q, self.V)
        uind = u_span - self.p
        denom = 0.0
        for l in range(self.q+1):
            temp = 0.0
            vind = v_span - self.q + l
            for k in range(self.p+1):
                temp += Bu[k] * self.G[uind+k][vind]
            denom += Bv[l] * temp
        return NIp * NJq * (self.P[I][J] - self(u, v)) / denom