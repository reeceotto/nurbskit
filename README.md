## NURBSKit

A Python-based toolkit for NURBS modelling and free-form deformation. NURBSKit 
supports Bezier, B-Spline and NURBS definitions for curves, surfaces 
and volumes. 

### Tools

The primary tools in NURBSKit are:

- curve, surface and volume evaluation
- analytical derivative evaluation
- analytical design sensitivity evaluation
- geometric transformation (scale, translation and rotation)
- point inversion and projection
- IGES and VTK surface export
- curve and surface interpolation and approximation
- visualisation interface using matplotlib
- parameterization and shape manipulation using free-form deformation

---

## Installation

The following command will clone the repository to `$HOME/nurbskit`.

```
git clone git@github.com:reeceotto/nurbskit.git nurbskit
```

After cloning, navigate to `$HOME/nurbskit` and add nurbskit as a Python package
using pip:

```
!pip install -e .
```

---

## Acknowledgements
Many of the algorithms used in NURBSKit are from 'The NURBS Book' by Piegl and
Tiller:

Les Piegl and Wayne Tiller. The NURBS Book. Monographs in Visual Communication. 
Springer, Berlin, Heidelberg, 2nd edition, 1997. ISBN 978-3-642-59223-2. 
DOI: 10.1007/978-3-642-59223-2.