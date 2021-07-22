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
        
def upper_triangular(A, b):
    """
    Algorithm for back substitution for resolving upper triangular linear system

    Parameters
    ----------
    A : bidimensional array
        Upper triangular matrix of coefficients. Must be a square matrix
    b : array
        Columns vector m-by-1 of known terms. 

    Returns
    -------
    x : array
        Column vector n-by-1 of solutions of the linear system.
    """
    [m,n] = shape(A)
    if m == n:
        x = zeros((n,1))
        if laplace(A) == 0: #det(A) != 0 iff A[i,i] != 0 for i in range(0, n-1)
            print('Warning: the matrix is singular\nUnable to solve the system of linear equation')
            return nan
        else:
            for i in range(m-1, -1, -1):
                sum = 0
                for j in range(i+1, n):
                    sum = sum + A[i,j]*x[j,0]   
                x[i,0] = (b[i]-sum)/A[i,i]
            return x
    else:
        print('Error: the matrix is not square\nUnable to solve the system of linear equation')
            
def lower_triangular(A, b):
    """
    Algorithm for forward substitution for resolving lower triangular linear system

    Parameters
    ----------
    A : bidimensional array
        Lower triangular matrix of coefficients. Must be a square matrix
    b : array
        Columns vector m-by-1 of known terms. 

    Returns
    -------
    x : array
        Column vector n-by-1 of solutions of the linear system.
    """
    [m,n] = shape(A)
    if m == n:
        x = zeros((n,1))
        if laplace(A) == 0: #det(A) != 0 iff A[i,i] != 0 for i in range(0, n-1)
            print('Warning: the matrix is singular\nUnable to solve the system of linear equation')
            return nan
        else:
            for i in range(0, m):
                sum = 0
                for j in range(0, i):
                    sum = sum + A[i,j]*x[j,0]   
                x[i,0] = (b[i]-sum)/A[i,i]
            return x
    else:
        print('Error: the matrix is not square\nUnable to solve the system of linear equation')
    
def lu_fact(A):
    """
    LU factorization of a square matrix.

    Parameters
    ----------
    A : bidimensional array
        Square, non-singular matrix.

    Returns
    -------
    L : bidimensional array
        Special lower triangular matrix.
    U : bidimensional array
        Upper triangular matrix extracted from A.

    """
    [m,n] = shape(A)
    if m == n:
        A = copy(A)
        L = identity(n)
        for k in range(0, n-1):
            if laplace(leading_principal_submatrix(A, k)) == 0:
                print('Warning: the matrix is singular\nUnable to decompose the matrix')
                return
            for i in range(k+1, n):
                mik = -A[i,k]/A[k,k]
                for j in range(k+1, n):
                    A[i,j] = A[i,j]+mik*A[k,j]
                L[i,k] = -mik
        U = triu(A)
        return L, U          
    else:
         print('Error: the matrix is not square\nUnable to decompose the matrix')
         return
       

#TEST UPPER TRIANGULAR LINEAR SYSTEM
# A = array([[1, -2, 3], [0, -10, 13], [0, 0, 1]])
# b = [1, 6, 2]
# x = [-1, 2, 2]

#TEST LOWER TRIANGULAR LINEAR SYSTEM
# A = array([[1, 0, 0], [-10, 13, 0], [1, -2, 3]])
# b = [2, 6, 1]
# x = [2, 2, 1]

#TEST SQUARE MATRICES
# A = array([[1, -2, 3], [4, -10, 13], [5, 1, 1]])

