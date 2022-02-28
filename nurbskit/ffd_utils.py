"""
Utility functions used in free-form deformation.

Author: Reece Otto 10/02/2022
"""
from nurbskit.utils import auto_knot_vector
from nurbskit.surface import BSplineSurface
import numpy as np

def auto_hull_2D(point_data, N_Px=4, N_Py=4, p=3, q=3, **kwargs):
    """
    Creates as a spline surface that embeds given point data.

    Arguments:
        point_data = list of 2D coordinates
    
    Keyword arguments:
        N_Px = number of control points in x direction
        N_Py = number of control points in y direction
        p = degree of hull in u direction
        q = degree of hull in v direction
        U = knot vector for u direction
        V = knot vector for v direction
    """
    # check if point data is 2D
    dims = np.nan * np.ones(len(point_data))
    for i in range(len(point_data)):
        dims[i] = len(point_data[i])
    if not np.all(dims == 2):
        raise AssertionError('Not all point data is 2D.')
    del dims

    # check that there are at least 2 control points in each direction
    if N_Px < 2 or N_Py < 2:
        raise AssertionError(f'At least two control points are required in \
            both the x and y directions.')

    # check that p is compatible with N_Px
    if p > N_Px - 1:
        str_1 = 'Degree of hull, p, is not compatible with' + \
        'given number of control points in direction x. \n'
        str_2 = f'Supplied path degree: {p}. \n'
        str_3 = f'Maximum allowed path degree: {N_Px-1}.'
        raise ValueError(str_1 + str_2 + str_3)

    # check that q is compatible with N_Py
    if q > N_Py - 1:
        str_1 = 'Degree of hull, q, is not compatible with' + \
        'given number of control points in direction y. \n'
        str_2 = f'Supplied path degree: {q}. \n'
        str_3 = f'Maximum allowed path degree: {N_Py-1}.'
        raise ValueError(str_1 + str_2 + str_3)

    # check that knot vector U is valid, if given
    if 'U' in kwargs:
        U = kwargs.get('U') 
        if len(U) != N_Px + p + 1:
            str_1 = 'Knot vector, U, is not compatible with N_Px and p. \n'
            str_2 = f'Supplied knot vector length: {len(U)}. \n'
            str_3 = f'Required knot vector length: {N_Px + p + 1}.'
            raise AttributeError(str_1 + str_2 + str_3)

    # check that knot vector V is valid, if given
    if 'V' in kwargs:
        V = kwargs.get('V') 
        if len(V) != N_Py + q + 1:
            str_1 = 'Knot vector, V, is not compatible with N_Py and q. \n'
            str_2 = f'Supplied knot vector length: {len(V)}. \n'
            str_3 = f'Required knot vector length: {N_Py + q + 1}.'
            raise AttributeError(str_1 + str_2 + str_3)

    # construct control point array
    point_xs = np.array(point_data)[:,0]
    min_x = point_xs.min()
    max_x = point_xs.max()

    point_ys = np.array(point_data)[:,1]
    min_y = point_ys.min()
    max_y = point_ys.max()

    P = np.nan * np.ones((N_Px, N_Py, 2))
    del_x = (max_x - min_x) / (N_Px - 1)
    del_y = (max_y - min_y) / (N_Py - 1)
    for i in range(N_Px):
        for j in range(N_Py):
            P[i][j][0] = min_x + i * del_x
            P[i][j][1] = min_y + j * del_y
    
    # construct knot vectors
    if 'U' not in kwargs:
        U = auto_knot_vector(N_Px, p)
    if 'V' not in kwargs:
        V = auto_knot_vector(N_Py, q)

    return BSplineSurface(P=P, p=p, q=q, U=U, V=V)
