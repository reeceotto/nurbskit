"""
Testing CAD export functions for surfaces.

Author: Reece Otto 11/02/2022
"""
from nurbskit.surface import NURBSSurface
from nurbskit.utils import auto_knot_vector
from nurbskit.cad_output import nurbs_surf_to_iges, surf_to_vtk

# generate NURBS surface
P = [[[0, 0, 0], [1, 0, 1], [2, 0, 0], [3, 0, -1]],
     [[0, 1, 1], [1, 1, 0], [2, 1, -1], [3, 1, 0]],
     [[0, 2, 0], [1, 2, -1], [2, 2, 0], [3, 2, 1]],
     [[0, 3, -1], [1, 3, 0], [2, 3, 1], [3, 3, 0]]]
G = [[1, 2, 1, 2],
     [2, 1, 2, 1],
     [1, 2, 1, 2],
     [2, 1, 2, 1]]
p = 3
q = 3
U = auto_knot_vector(len(P), p)
V = auto_knot_vector(len(P[0]), q)
surf = NURBSSurface(P=P, G=G, p=3, q=3, U=U, V=V)

# export surface as IGES
nurbs_surf_to_iges(surf, file_name='nurbs_surf')

# export surface as vtk
surf_to_vtk(surf, file_name='nurbs_surf')

