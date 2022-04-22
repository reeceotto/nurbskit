"""
Point inversion toolkit.

The process of point inversion:
    - specify the Cartesian coordinates of a point (x,y,z) that is known to 
      lie on a path, surface or volume
    - calculate the location of this point with respect to the parametric
      space of the path, surface or volume (u,v,w)

Author: Reece Otto 18/09/2020
"""
import numpy as np

def point_proj_path(path, xyz_point, n_param_guesses=3, max_sub_it=10, 
    tol_dist=1E-3, tol_cos=1E-3, tol_end=1E-3):
    # root function
    def f(path, xyz_point, u):
        return np.dot(path.dCdu_fd(u), path(u) - xyz_point)

    # derivative of root function
    def fu(path, xyz_point, u):
        return np.dot(path.d2Cdu2_fd(u), path(u) - xyz_point) + \
               np.linalg.norm(path.dCdu_fd(u))**2

    # define distance function
    def distance(path, xyz_point, u):
        return np.linalg.norm(path(u) - xyz_point)

    # define zero cosine function
    def zero_cos(path, xyz_point, u):
        return np.linalg.norm(f(path, xyz_point, u)) / \
        np.linalg.norm(path.dCdu_fd(u)) / np.linalg.norm(path(u) - xyz_point)

    # define end-point projection detector function
    def end_proj(path, xyz_point, u_old, u_new):
        return np.linalg.norm((u_new - u_old) * path.dCdu_fd(u_old))

    # establish list of initial guesses for u
    us = np.linspace(path.u_start(), path.u_end(), n_param_guesses)
    dists = np.nan * np.ones(n_param_guesses)
    u_projs = np.nan * np.ones(n_param_guesses)
    for i in range(len(us)):
        # set initial guess for Newton iteration
        u_old = us[i]
        u_new = us[i]

        # initialise iteration index
        j = 0

        # perform Newton iterations until Euclidean distance is within tolerance 
        # OR there is zero cosine between xyz_point and projected point
        while (distance(path, xyz_point, u_old) > tol_dist) and \
        (zero_cos(path, xyz_point, u_old) > tol_cos) and \
        j < max_sub_it:
            # compute new parameter value
            u_new = u_old - f(path, xyz_point, u_old) / fu(path, xyz_point, u_old)

            # ensure parameter value stays on path
            if not np.array_equal(path.P[0], path.P[-1]):
                if u_new < path.u_start(): u_new = path.u_start()
                if u_new > path.u_end(): u_new = path.u_end()
            else:
                if u_new < path.u_start(): 
                    u_new = path.u_end() - path.u_start() + u_new
                if u_new > path.u_end():
                    u_new = path.u_start() + u_new - path.u_end()

            # check if point is being projected to an end-point
            if end_proj(path, xyz_point, u_old, u_new) < tol_end:
                dists[i] = distance(path, xyz_point, u_new)
                u_projs[i] = u_new
                break
        
            # reset parameter for next iteration
            u_old = u_new

            # add to iteration index for next iteration
            j += 1

        dists[i] = distance(path, xyz_point, u_new)
        u_projs[i] = u_new

    # find which u value resulted in the minimum distance to the curve
    ind = np.argmin(dists)
    return u_projs[ind]

def point_inv_surf(surface, xyz_point, tol=1E-3):
    """
    Point inversion algorithm from Piegl and Tiller but with removed
    point projection functionality.
    Changes:
        - removed convergence criteria 2 and 4
        - added functionality for non-clamped curves
    
    See Section 6.1 of:
    'The NURBS Book' - Les Piegl & Wayne Tiller, 1997.
    
    Arguments:
        surface = surface object
        xyz_point = Cartesian point that exists on surface

    Keyword arguments:
        tol = convergence tolerance based on Euclidean distance
    """
    # defining root functions and their derivatives
    def f(surface, xyz_point, u, v):
        return np.dot(surface.dSdu(u, v), 
                      np.array(surface(u, v)) - np.array(xyz_point))
    
    def fu(surface, xyz_point, u, v):
        return np.dot(surface.dSdu(u, v), surface.dSdu(u, v)) + \
               np.dot(np.array(surface(u, v)) - np.array(xyz_point), 
                      surface.d2Sdu2(u, v))
    
    def fv(surface, xyz_point, u, v):
        return np.dot(surface.dSdu(u, v), surface.dSdv(u, v)) + \
               np.dot(np.array(surface(u, v)) - np.array(xyz_point), 
                                       surface.d2Sdudv(u, v))
    
    def g(surface, xyz_point, u, v):
        return np.dot(surface.dSdv(u, v), 
                      np.array(surface(u, v)) - np.array(xyz_point))
    
    def gu(surface, xyz_point, u, v):
        return np.dot(surface.dSdu(u, v), surface.dSdv(u, v)) + \
               np.dot(np.array(surface(u, v)) - np.array(xyz_point), 
                                       surface.d2Sdudv(u, v)) 
    
    def gv(surface, xyz_point, u, v):
        return np.dot(surface.dSdv(u, v), surface.dSdv(u, v)) + \
               np.dot(np.array(surface(u, v)) - np.array(xyz_point), 
                      surface.d2Sdu2(u, v))
    
    def J(surface, xyz_point, u, v):
        return np.array([[fu(surface, xyz_point, u, v), 
            fv(surface, xyz_point, u, v)],
            [gu(surface, xyz_point, u, v), 
            gv(surface, xyz_point, u, v)]])
    
    def kappa(surface, xyz_point, u, v):
        return np.array([-f(surface, xyz_point, u, v),
                         -g(surface, xyz_point, u, v)])
    
    # initial guess for Newton iteration
    params_old = np.array([0.5, 0.5])
    params_new = params_old.copy()
    
    # define distance function
    def distance(surface, xyz_point, u, v):
        rel_vec = np.array(surface(u, v)) - np.array(xyz_point)
        return np.linalg.norm(rel_vec)

    # perform Newton iterations until Euclidean distance is with in tolerance
    while distance(surface, xyz_point, params_old[0], params_old[1]) > tol:    
        
        # calculate params for next iteration
        delta = np.linalg.solve(J(surface, xyz_point, params_old[0], 
            params_new[1]), kappa(surface, xyz_point, params_old[0], 
            params_new[1]))
        params_new = params_old + delta

        # ensure the parameters stay on the surface
        if params_new[0] < surface.u_start():
            params_new[0] = surface.u_start()
        elif params_new[0] > surface.u_end():
            params_new[0] = surface.u_end()
        
        if params_new[1] < surface.v_start():
            params_new[1] = surface.v_start()
        elif params_new[1] > surface.v_end():
            params_new[1] = surface.v_end()

        # reset u for next iteration
        params_old = params_new
        
    return params_new

def PtInvPTCustVol(volume, cartPoint, tol=1E-3):
    """
    Point inversion algorithm adapted from Piegl and Tiller but with removed
    point projection functionality.
    Changes:
        - removed convergence criteria 2 and 4
        - added functionality for non-clamped curves
    
    This algorithm was adapted from section 6.1 of:
    'The NURBS Book' - Les Piegl & Wayne Tiller, 1997.
    
    volume = volume object
    cartPoint = Cartesian point to be projected
    tol = tolerance for Euclidean distance
    """
    # defining root functions and their derivatives
    def r(u, v, w):
        return np.array(volume(u, v, w)) - np.array(cartPoint)
    
    def f(u, v, w):
        return np.dot(r(u, v, w), volume.dVdu(u, v, w))
    
    def fu(u, v, w):
        return np.dot(volume.dVdu(u, v, w), volume.dVdu(u, v, w)) + \
               np.dot(r(u, v, w), volume.d2Vdu2(u, v, w))
    
    def fv(u, v, w):
        return np.dot(volume.dVdv(u, v, w), volume.dVdu(u, v, w)) + \
               np.dot(r(u, v, w), volume.d2Vdudv(u, v, w))
    def fw(u, v, w):
        return np.dot(volume.dVdw(u, v, w), volume.dVdu(u, v, w)) + \
               np.dot(r(u, v, w), volume.d2Vdudw(u, v, w))
    
    def g(u, v, w):
        return np.dot(r(u, v, w), volume.dVdv(u, v, w))
    
    def gu(u, v, w):
        return np.dot(volume.dVdu(u, v, w), volume.dVdv(u, v, w)) + \
               np.dot(r(u, v, w), volume.d2Vdudv(u, v, w))
    
    def gv(u, v, w):
        return np.dot(volume.dVdv(u, v, w), volume.dVdv(u, v, w)) + \
               np.dot(r(u, v, w), volume.d2Vdv2(u, v, w))
    
    def gw(u, v, w):
        return np.dot(volume.dVdw(u, v, w), volume.dVdv(u, v, w)) + \
               np.dot(r(u, v, w), volume.d2Vdvdw(u, v, w))
    
    def h(u, v, w):
        return np.dot(r(u, v, w), volume.dVdw(u, v, w))
    
    def hu(u, v, w):
        return np.dot(volume.dVdu(u, v, w), volume.dVdw(u, v, w)) + \
               np.dot(r(u, v, w), volume.d2Vdudw(u, v, w))
    
    def hv(u, v, w):
        return np.dot(volume.dVdv(u, v, w), volume.dVdw(u, v, w)) + \
               np.dot(r(u, v, w), volume.d2Vdvdw(u, v, w))
    
    def hw(u, v, w):
        return np.dot(volume.dVdw(u, v, w), volume.dVdw(u, v, w)) + \
               np.dot(r(u, v, w), volume.d2Vdw2(u, v, w))
    
    def J_det(u, v, w):
        return fu(u, v, w) * gv(u, v, w) * hw(u, v, w) + \
               fv(u, v, w) * gw(u, v, w) * hu(u, v, w) + \
               fw(u, v, w) * gu(u, v, w) * hv(u, v, w) - \
               hu(u, v, w) * gv(u, v, w) * fw(u, v, w) - \
               hv(u, v, w) * gw(u, v, w) * fu(u, v, w) - \
               hw(u, v, w) * gu(u, v, w) * fv(u, v, w)
               
    def J_inv(u, v, w):
        return 1/(J_det(u, v, w)) * np.array([[gv(u,v,w)*hw(u,v,w) - gw(u,v,w)*hv(u,v,w), fw(u,v,w)*hv(u,v,w) - fv(u,v,w)*hw(u,v,w), fv(u,v,w)*gw(u,v,w) - fw(u,v,w)*gv(u,v,w)],
                                              [gw(u,v,w)*hu(u,v,w) - gu(u,v,w)*hw(u,v,w), fu(u,v,w)*hw(u,v,w) - fw(u,v,w)*hu(u,v,w), fw(u,v,w)*gu(u,v,w) - fu(u,v,w)*gw(u,v,w)],
                                              [gu(u,v,w)*hv(u,v,w) - gv(u,v,w)*hu(u,v,w), fv(u,v,w)*hu(u,v,w) - fu(u,v,w)*hv(u,v,w), fu(u,v,w)*gv(u,v,w) - fv(u,v,w)*gu(u,v,w)]])
    
    def kappa(u, v, w):
        return np.array([-f(u, v, w),
                         -g(u, v, w),
                         -h(u, v, w)])
    
    # initial guess for Newton iteration
    params_old = np.array([0.5, 0.5, 0.5])
    params_new = params_old.copy()
    
    # perform Newton iterations until Euclidean distance is with in tolerance
    while np.linalg.norm(r(params_old[0], params_old[1], params_old[2])) > tol:
        # calculate params for next iteration
        
        delta = J_inv(params_old[0], params_old[1], params_old[2]).dot(kappa(params_old[0], params_old[1], params_old[2]))
        params_new = params_old + delta
        
        # ensure params stay in/on volume
        if params_new[0] < volume.u_start():
            params_new[0] = volume.u_start()
        elif params_new[0] > volume.u_end():
            params_new[0] = volume.u_end()
        
        if params_new[1] < volume.v_start():
            params_new[1] = volume.v_start()
        elif params_new[1] > volume.v_end():
            params_new[1] = volume.v_end()
            
        if params_new[2] < volume.w_start():
            params_new[2] = volume.w_start()
        elif params_new[2] > volume.w_end():
            params_new[2] = volume.w_end()
        
        # reset u for next iteration
        params_old = params_new
    print(params_new)
    return params_new