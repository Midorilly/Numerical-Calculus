# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 00:40:48 2021

@author: angel
"""
from numpy import ones, size, linalg, copy, shape, nan, transpose, dot
from linear_algebra import gauss_elimination, rank

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
    
def polynomial(x, y):
    """
    Computes coefficients of the interpolation polynomial.

    Parameters
    ----------
    x : array
        Array of variables, n-by-1.
    y : array
        Array of knwon terms, n-by-1.

    Returns
    -------
    a : array
        Array of coefficients.

    """
    
    var = copy(x)
    known = copy(y)
    V = vandermonde_matrix(var)
    a = gauss_elimination(V, known)
    return a
    
def linear_least_squares(M, v):
    """
    Solves the linear least squares problem.
    If rank(A) is its maximum then the linear least squares problem has one unique solution, obtained
    as solution of a n-equations, n-variables system AtAx = Atb, known as normal system
    Parameters
    ----------
    A : bidimensional array
        Matrix of coefficients. 
    b : array
        Columns vector m-by-1 of known terms. 

    Returns
    -------
    x : array
        Column vector n-by-1 of solutions of the linear system.

    """
   
    B = copy(M)
    [m,n] = shape(B)
    if rank(B) != min(m,n):
        print('Warning: can not be solved since the rank of the matrix is not its maximum value')
        return nan
    else:
        
        A = copy(M)
        At = transpose(M)
        b = copy(v)
        b = transpose(b)
        
        AtA = dot(At, A)
        Atb = transpose(dot(At, b))
        print(AtA, Atb)
        
        x = gauss_elimination(AtA, Atb)
        print('x*:')
        return x

#SQUARE_LINEAR_SYSTEM
#A = array([[1, 1, 0], [2, 1, 1], [3, 0, 1]], dtype=float)
#b = array([[1,1,1]], dtype=float)
#x* = [0.5, 0.5, -0.5]
#A = array([[-1, 0, -2], [2,1,2], [-1,0,-2], [2,1,1], [3,1,3]], dtype=float)
#b = array([[1,1,1,1,1]], dtype=float)
#x* = [0, 9/5, -2/5]