# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 01:08:56 2021

@author: angel
"""

from numpy import nan, zeros, copy, identity, triu, shape, delete, array, int16

def transposition(A):
    """
    Transpose of a m-by-n matrix in a n-by-m matrix.

    Parameters
    ----------
    A : bidimensional array
        Matrix.

    Returns
    -------
    T : bidimensional array
        Transposed matrix.
    """
    [m,n] = shape(A)
    T = zeros((n,m))
    for i in range(0,m):
        for j in range(0,n):
            T[j,i] = A[i,j]
    return T

def leading_principal_submatrix(A, k):
    """
    Leading principal sumbatrix of A only considers the first k rows and k columns of A

    Parameters
    ----------
    A : bidimensional array
        Matrix.
    k : int
        Number of rows and columns of A which remain.

    Returns
    -------
    A : bidimensional array
        Leading principal sumbatrix.
    """
    [m,n] = shape(A)
    if k>m:
        print('Error: the rows of the matrix are less than k\nChoose a different k')
    elif k>n:
        print('Error: the columns of the matrix are less than k\nChoose a different k')
    else:    
        for i in range(m-1,k-1,-1):
            #print(i)
            A = delete(A, i, axis=0)        
        for j in range(n-1,k-1,-1):
            #print(j)
            A = delete(A, j, axis=1)
        return A

def laplace(A):
    """
    Laplace expansion for computing the determinant of a square matrix.
    Progression along the first row.

    Parameters
    ----------
    A : bidimensional array
        Matrix. Must be a square matrix in order to calculate its determinant.

    Returns
    -------
    d : float
        Determinant of A.
    """
    [m,n] = shape(A)
    if m == n:
        if n == 1:
            d = A[0,0]
        else:
            d = 0
            for j in range(0,n):
                A1j = delete(A, 0, axis=0)
                A1j = delete(A1j, j, axis=1)
                d = d + (-1)**(j) * A[0,j] * laplace(A1j)
        return d            
    else:
        print('Warning: The input matrix is not square\nUnable to compute its determinant')
    