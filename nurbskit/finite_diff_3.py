"""
Finite difference functions for tri-variate vector functions.
TODO: remove this script

Author: Reece Otto 13/10/2021
"""

def fd_3(f, u, v, w, h, ind):
    if ind == 0:
        der = (f(u + h, v, w) - f(u, v, w)) / h
    elif ind == 1:
        der = (f(u, v + h, w) - f(u, v, w)) / h
    elif ind == 2:
        der = (f(u, v, w + h) - f(u, v, w)) / h
    return der

def bd_3(f, u, v, w, h, ind):
    if ind == 0:
        der = (f(u - h, v, w) - f(u, v, w)) / h
    elif ind == 1:
        der = (f(u, v - h, w) - f(u, v, w)) / h
    elif ind == 2:
        der = (f(u, v, w - h) - f(u, v, w)) / h
    return der

def cd_3(f, u, v, w, h, ind):
    if ind == 0:
        der = (f(u + h, v, w) - f(u - h, v, w)) / 2 / h
    elif ind == 1:
        der = (f(u, v + h, w) - f(u, v - h, w)) / 2 / h
    elif ind == 2:
        der = (f(u, v, w + h) - f(u, v, w - h)) / 2 / h
    return der