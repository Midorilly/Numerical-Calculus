# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 00:40:48 2021

@author: angel
"""
from numpy import ones, size, linalg

def vandermonde_matrix(x):
    """
    Computes a Vandermonde matrix given an array x of values.

    Parameters
    ----------
    x : array
        Array of values.

    Returns
    -------
    V : bidimensional array
        Vandermonde matrix.

    """
    m = size(x) 
    n = m+1
    V = ones((m, n))
    for j in range(1, n):
        for i in range(0, m):
            V[i,j] = pow(x[i],j)  
    return V

def vandermonde_cond(x):
    
    V = vandermonde_matrix(x)
    print(linalg.cond(V))
