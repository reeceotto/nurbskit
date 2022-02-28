"""
Spline-based volume classes.
Note: this script does not currently work.

Author: Reece Otto 12/10/2021
"""
import numpy as np
from .utils import FindSpan, BSplineBasisFuns, WeightedControlPoints
from .finite_diff_3 import fd_3, bd_3, cd_3

class Volume():
    """
    Base class for all volume/solid objects.
    
    u, v, w = parametric coordinates
    """
    
    def __repr__(self):
        pass
    
    def __call__(self, u, v, w):
        pass
    
    # volume derivatives
    def dVdu(self, u, v, w):
        # Calculates dV/du using finite differences.
        du = 0.001
        V0 = self(u, v, v)
        # backwards difference if near u end
        if u + du > self.U[-(self.p + 1)]:
            der = (V0 - self(u - du, v, w)) / du
        # forward difference if near u start
        elif u - du < self.U[self.p]:
            der = (self(u + du, v, w) - V0) / du
        # central difference if anywhere else
        else:
            der = (self(u + du, v, w) - self(u - du, v, w)) / (2.0 * du)
        return der
    
    def dVdv(self, u, v, w):
        # Calculates dV/dv using finite differences.
        dv = 0.001
        V0 = self(u, v, v)
        # backwards difference if near v end
        if v + dv > self.V[-(self.q + 1)]:
            der = (V0 - self(u, v - dv, w)) / dv
        # forward difference if near v start
        elif v - dv < self.V[self.q]:
            der = (self(u, v + dv, w) - V0) / dv
        # central difference if anywhere else
        else:
            der = (self(u, v + dv, w) - self(u, v - dv, w)) / (2.0 * dv)
        return der
    
    def dVdw(self, u, v, w):
        # Calculates dV/dw using finite differences.
        dw = 0.001
        V0 = self(u, v, v)
        # backwards difference if near w end
        if w + dw > self.W[-(self.r + 1)]:
            der = (V0 - self(u, v, w - dw)) / dw
        # forward difference if near w start
        elif w - dw < self.W[self.r]:
            der = (self(u, v, w + dw) - V0) / dw
        # central difference if anywhere else
        else:
            der = (self(u, v, w + dw) - self(u, v, w - dw)) / (2.0 * dw)
        return der
    
    def d2Vdudv(self, u, v, w):
        # Calculates d^2V/dudv or d^2/dvdu using finite differences.
        dv = 0.001
        def dVdu(u, v, w):
            return self.dVdu(u, v, w)
        # backwards difference if near v end
        if v + dv > self.V[-(self.q + 1)]:
            der = bd_3(dVdu, u, v, w, dv, ind=1)
        # forward difference if near v start
        elif v - dv < self.V[self.q]:
            der = fd_3(dVdu, u, v, w, dv, ind=1)
        # central difference if anywhere else
        else:
            der = cd_3(dVdu, u, v, w, dv, ind=1)
        return der
        
    def d2Vdudw(self, u, v, w):
        # Calculates d^2V/dudw or d^2/dwdu using finite differences.
        dw = 0.001
        def dVdu(u, v, w):
            return self.dVdu(u, v, w)
        # backwards difference if near w end
        if w + dw > self.W[-(self.r + 1)]:
            der = bd_3(dVdu, u, v, w, dw, ind=2)
        # forward difference if near w start
        elif w - dw < self.W[self.r]:
            der = fd_3(dVdu, u, v, w, dw, ind=2)
        # central difference if anywhere else
        else:
            der = cd_3(dVdu, u, v, w, dw, ind=2)
        return der
    
    def d2Vdvdw(self, u, v, w):
        # Calculates d^2V/dvdw or d^2/dvdw using finite differences.
        dw = 0.001
        def dVdv(u, v, w):
            return self.dVdv(u, v, w)
        # backwards difference if near w end
        if w + dw > self.W[-(self.r + 1)]:
            der = bd_3(dVdv, u, v, w, dw, ind=2)
        # forward difference if near w start
        elif w - dw < self.W[self.r]:
            der = fd_3(dVdv, u, v, w, dw, ind=2)
        # central difference if anywhere else
        else:
            der = cd_3(dVdv, u, v, w, dw, ind=2)
        return der
    
    def d2Vdu2(self, u, v, w):
        # Calculates d^2V/du^2 using finite differences.
        du = 0.001
        V0 = self(u, v, v)
        # backwards difference if near u end
        if u + du > self.U[-(self.p + 1)]:
            der = (V0 - 2 * self(u - du, v, w) + self(u - 2 * du, v, w)) / (du**2)
        # forward difference if near u start
        elif u - du < self.U[self.p]:
            der = (self(u + 2 * du, v, w) - 2*self(u + du, v, w) + V0) / (du**2)
        # central difference if anywhere else
        else:
            der = (self(u + du, v, w) - 2 * V0 + self(u - du, v, w)) / (du**2)
        return der
    
    def d2Vdv2(self, u, v, w):
        # Calculates d^2V/dv^2 using finite differences.
        dv = 0.001
        V0 = self(u, v, v)
        # backwards difference if near v end
        if v + dv > self.V[-(self.q + 1)]:
            der = (V0 - 2 * self(u, v - dv, w) + self(u, v - 2 * dv, w)) / (dv**2)
        # forward difference if near v start
        elif v - dv < self.V[self.q]:
            der = (self(u, v + 2 * dv, w) - 2*self(u, v + dv, w) + V0) / (dv**2)
        # central difference if anywhere else
        else:
            der = (self(u, v + dv, w) - 2 * V0 + self(u, v - dv, w)) / (dv**2)
        return der
    
    def d2Vdw2(self, u, v, w):
        # Calculates d^2V/dw^2 using finite differences.
        dw = 0.001
        V0 = self(u, v, v)
        # backwards difference if near w end
        if w + dw > self.W[-(self.r + 1)]:
            der = (V0 - 2 * self(u, v, w - dw) + self(u, v, w - 2 * dw)) / (dw**2)
        # forward difference if near w start
        elif w - dw < self.W[self.r]:
            der = (self(u, v, w + 2 * dw) - 2*self(u, v, w + dw) + V0) / (dw**2)
        # central difference if anywhere else
        else:
            der = (self(u, v, w + dw) - 2 * V0 + self(u, v, w - dw)) / (dw**2)
        return der
    
    def u_start(self):
        # Calculates parametric starting point of volume in u direction.
        return self.U[self.p]
    
    def u_end(self):
        # Calculates parametric ending point of volume in u direction.
        return self.U[-(self.p + 1)]
    
    def v_start(self):
        # Calculates parametric starting point of volume in v direction.
        return self.V[self.q]
    
    def v_end(self):
        # Calculates parametric ending point of volume in v direction.
        return self.V[-(self.q + 1)]
    
    def w_start(self):
        # Calculates parametric starting point of volume in v direction.
        return self.W[self.r]
    
    def w_end(self):
        # Calculates parametric ending point of volume in v direction.
        return self.W[-(self.r + 1)]
    
    def Knots(self):
        # returns flat list with Cartesian coordinates of knots
        knots = []
        for i in self.U:
            for j in self.V:
                for k in self.W:
                    knots.append(self(i, j, k))
        return knots
    
class NURBSVolume(Volume):
    """
    NURBS volume object.
    
    P = control points
    G = control point weights
    p, q, r = orders of curves
    U, V, W = knot vectors
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
    def __repr__(self):
        return f"NURBSVolume(P={self.P}, G={self.G}, p={self.p}, q={self.q},\
                 r={self.r}, U={self.U}, V={self.V}, W={self.W})"
    
    def __call__(self, u, v, w, **kwargs):
        # Algorithm adapted from NURBS surface algorithm
        Pw = WeightedControlPoints(self.P, self.G, dimension=3)
        u_span = FindSpan(self.p, u, self.U)
        Bu = BSplineBasisFuns(u_span, u, self.p, self.U)
        v_span = FindSpan(self.q, v, self.V)
        Bv = BSplineBasisFuns(v_span, v, self.q, self.V)
        w_span = FindSpan(self.r, w, self.W)
        Bw = BSplineBasisFuns(w_span, w, self.r, self.W)
        temp1 = np.nan * np.ones((self.q + 1, self.r + 1), dtype=np.ndarray)
        for m in range(self.r + 1):
            for l in range(self.q + 1):
                temp1[l][m] = 0
                for k in range(self.p + 1):
                    temp1[l][m] += Bu[k] * np.array(Pw[u_span-self.p+k]
                                                      [v_span-self.q+l]
                                                      [w_span-self.r+m])
        temp2 = np.nan * np.ones(self.r + 1, dtype=np.ndarray)
        for m in range(self.r + 1):
            temp2[m] = 0
            for l in range(self.q + 1):
                temp2[m] += Bv[l] * temp1[l][m]
        Vw = 0
        for m in range(self.r+1):
            Vw += Bw[m] * temp2[m]
        V = np.zeros(len(Vw)-1)
        for k in range(len(V)):
            V[k] = Vw[k] / Vw[-1]
        return V