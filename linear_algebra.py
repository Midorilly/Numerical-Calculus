# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 01:08:56 2021

@author: angel
"""

from numpy import dot, set_printoptions, transpose, nan, zeros, copy, identity, triu, shape, delete, array, hstack, vstack
from numpy.linalg import inv

set_printoptions(precision=4)

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

def leading_principal_submatrix(M, k):
    """
    Leading principal sumbatrix of A only considers the first k rows and k columns of A

    Parameters
    ----------
    A : bidimensional array.
        Matrix.
    k : int.
        Number of rows and columns of A which remain.

    Returns
    -------
    A : bidimensional array.
        Leading principal sumbatrix.
    """
    A = copy(M)
    [m,n] = shape(A)
    if k>m:
        print('Error: the rows of the matrix are less than k\nChoose a different k')
    elif k>n:
        print('Error: the columns of the matrix are less than k\nChoose a different k')
    else:    
        for i in range(m-1,k-1,-1):
            A = delete(A, i, axis=0)        
        for j in range(n-1,k-1,-1):
            A = delete(A, j, axis=1)
        return A

def laplace(M):
    """
    Laplace expansion for computing the determinant of a square matrix.
    Progression along the first row.

    Parameters
    ----------
    A : bidimensional array
        Matrix. Must be a square matrix in order to calculate its determinant.

    Returns
    -------
    det : float
        Determinant of A.
    """
    tol=1e-10
    A = copy(M)
    [m,n] = shape(A)
    if abs(m-n)<tol:
        if abs(n-1)<tol:
            det = A[0,0]
        else:
            det = 0
            for j in range(0,n):
                A1j = delete(A, 0, axis=0)
                A1j = delete(A1j, j, axis=1)
                det = det + (-1)**(j) * A[0,j] * laplace(A1j)
        return det            
    else:
        print('Warning: The input matrix is not square\nUnable to compute its determinant')
        
def rank(A):
    """
    Rank of a m-by-n matrix, computed counting non-null rows of its echelon form.
    The rank of a matrix can be defined in different equivalent ways:
        -  largest order of any non-zero minor in A
        -  dimension of the vector space generated (or spanned) by its columns
        -  maximal number of linearly independent columns of A
        -  dimension of the vector space spanned by its rows
        -  measure of the "nondegenerateness" of the system of linear equations 
           and linear transformation encoded by A

    Parameters
    ----------
    A : bidimensional array
        Matrix.

    Returns
    -------
    rank : int
        DESCRIPTION.

    """
    M = echelon_form(A)        
    [m,n] = shape(M)
    rank = 0
    for i in range(m):
        if any((M[i,:] != 0)): #counts non-null rows
            rank = rank + 1            
    return rank        
        
def upper_triangular(M, v):
    """
    Algorithm for back substitution for resolving upper triangular linear system

    Parameters
    ----------
    A : bidimensional array
        Upper triangular matrix of coefficients. Must be a square matrix
    b : array
        Columns vector n-by-1 of known terms. 

    Returns
    -------
    x : array
        Column vector n-by-1 of solutions of the linear system.
    """
    tol=1e-10
    A = copy(M)
    b = copy(v)
    [m,n] = shape(A)
    if abs(m-n)<tol:
        x = zeros((n))
        if abs(laplace(A))<tol: #det(A) != 0 iff A[i,i] != 0 for i in range(0, n-1)
            print('Warning: the matrix is singular\nUnable to solve the system of linear equation')
            return nan
        else:
            for i in range(m-1, -1, -1):
                sum = 0
                for j in range(i+1, n):
                    sum = sum + A[i,j]*x[j]   
                x[i] = (b[0,i]-sum)/A[i,i]
                
            return x
    else:
        print('Error: the matrix is not square\nUnable to solve the system of linear equation')
            
def lower_triangular(M, v):
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
    tol=1e-10
    A = copy(M)
    b = copy(v)
    [m,n] = shape(A)
    if abs(m-n)<tol:
        x = zeros((n,1))
        if abs(laplace(A))<tol: #det(A) != 0 iff A[i,i] != 0 for i in range(0, n-1)
            print('Warning: the matrix is singular\nUnable to solve the system of linear equations')
            return nan
        else:
            for i in range(0, m):
                sum = 0
                for j in range(0, i):
                    sum = sum + A[i,j]*x[j,0]   
                x[i,0] = (b[i]-sum)/A[i,i] #DA RIVEDERE
            return x
    else:
        print('Error: the matrix is not square\nUnable to solve the system of linear equations')

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
    tol=1e-15
    [m,n] = shape(A)
    if abs(m-n)<tol:
        A = copy(A)
        L = identity(n)
      
        for k in range(0, n-1):
            if abs(A[k,k])<tol:
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
     
def echelon_form(M):
    """
    Transforms a matrix to its echelon form.

    Parameters
    ----------
    M : bidimensional array.
        Matrix.

    Returns
    -------
    Echelon form matrix

    """
    # Base case
    # if r == 0 or c == 0 -> M is already in its echelon form
    # if M.r == 1 or M.c == 1 -> one remaining element
    r, c = M.shape
    if (r == 0 or c == 0) or (r == 1 and c == 1):
        return M

    # Looks for the first non-null element in the column. If not found, enters the else
    for i in range(len(M)):
        if M[i,0] != 0:
            break
    else:
        # If every element in the first column are null, recursively calls the function on the second column
        MR = echelon_form(M[:,1:])
        # Prendiamo il risultato della chiamata e riaggiungiamo la colonna rimossa
        return hstack([M[:,:1], MR])

    # Cambio riga nel caso di elemento diverso da zero presente in un'altra colonna
    if i > 0:
        ith_row = M[i].copy()
        M[i] = M[0]
        M[0] = ith_row

    # Calcoliamo lambda da moltiplicare alla colonna affinché sommata venga annullata
    pivot = M[0,0]
    coeff = - M[0] / pivot
    M[1:] = coeff * M[1:,0:1] + M[1:]  

    # Chiamata ricorsiva eliminando la riga e la colonna coinvolta 
    MR = echelon_form(M[1:,1:])

    # Riaggiungiamo la riga e la colonna rimossa al risultato dell'operazione
    return vstack([M[:1], hstack([M[1:,:1], MR])])        
    
def inverse_linear_system(M, v):
    """
    Solves a linear system when A is a square, non-singular matrix computing x = A^-1*b

    Parameters
    ----------
    A : bidimensional array.
        Square, non-singular matrix.
    b : array
        Columns vector n-by-1 of known terms. 

    Returns
    -------
    x : array
        Column vector n-by-1 of solutions of the linear system.

    """
    A = copy(M)
    b = transpose(copy(v))
    tol=1e-10
    if abs(laplace(A))>=tol:
        Ai = inv(A)
        print(Ai)
        x = dot(Ai,b)
        return x
    else: 
        print('Warning: the matrix is singular\nUnable to solve the system of linear equations')
        return nan
    
def lu_linear_system(M, v):
    """
    Solves a linear system when A is a square matrix computing its LU factorization. 

    Parameters
    ----------
    A : bidimensional array.
        Square matrix.
    b : array
        Columns vector n-by-1 of known terms. 

    Returns
    -------
    x : array
        Column vector n-by-1 of solutions of the linear system.

    """
    A = copy(M)
    b = transpose(copy(v))
    L, U = lu_fact(A)
    print(L)
    print(U)
    
    y = lower_triangular(L, b)
    x = upper_triangular(U, y)
       
    return x
    
def gauss_elimination(M, v):
    """
    Implements Gauss elimination algorithm to solve m-equations, n-variable systems.

    Parameters
    ----------
    A : bidimensional array
        Matrix of coefficients, m-by-n. 
    b : array
        Columns vector m-by-1 of known terms. 

    Returns
    -------
    x : array
        Vector n-by-1 of variables.

    """
    B = copy(M)
    d = copy(v)
    
    consistency, solution = is_consistent(M, v)
    if(consistency):
        parameter = []
        for i in range(0, solution):
            param = float(input('Please type the chosen real value for parameter'))
            parameter.append(param)
            i = i+1
           
        d = transpose(d)
        
        #creates complete matrix
        complete = hstack((B, d))
        echelon = echelon_form(complete)
        
        #verifies if the echelon form contains any null row and deletes it
        [m,n] = shape(echelon)
        i = 0
        for i in range(m):
            if all((echelon[i,:] == 0)): 
                echelon = delete(echelon, i, axis=0)
                m = m-1
      
        
        #separates the matrix of coefficients and the array of known terms to perform the 
        #back sostituition algorithm and obtain an array of variables
        [m,n] = shape(echelon)
        known = array([echelon[:, n-1]], dtype=float)
        coefficient = delete(echelon, n-1, axis=1)
        
        #determines if the coefficient matrix is square; if not, parametrization is applied
        [k,l] = shape(coefficient)
        if k != l: #parametrization          
            p_row, p_col = pivot_index(coefficient)
            i = 0
            j = 0
            p = 0
            for p in range(solution):
                for j in range(l):
                    if j in p_col:  
                        j = j+1
                    else:
                        for i in range(k):                       
                            known[0, i] = known[0, i] - coefficient[i,j]*parameter[p]  
                        coefficient = delete(coefficient, j, axis=1)   
                        j = j+1
                        
            x = upper_triangular(coefficient, known)
            
        else:    
            x = upper_triangular(coefficient, known)
        
        return x
    
def pivot_index(M):
    """
    Parameters
    ----------
    M : bidimensional array
        Matrix in echelon form.

    Returns
    -------
    pivot: array
        Array of pivot columns indexes.
    """
    [m,n] = shape(M)
    pivot_row = []
    pivot_column = []
    i = 0
    j = 0
    for i in range(m):
        for j in range(n):
            if M[i,j] != 0:
                pivot_row.append(i)
                pivot_column.append(j)
                break
    return pivot_row, pivot_column

def is_consistent(M, v):
    """
    Determines whether a linear system is consistent (has at least a solution) or not. A linear system
    is consistent when the rank of matrix of coefficients and the rank of the complete matrix has
    the same value (Rouché-Capelli theorem)

    Parameters
    ----------
    A : bidimensional array.
        Matrix of coefficients, m-by-n
    b : array.
        Columns vector m-by-1 of known terms.

    Returns
    -------
    is_consistent : boolean.

    """
    A = copy(M)
    b = copy(v)
    [m,n] = shape(A)
    b = transpose(b)
    is_consistent = False
    #computes incomplete matrix rank
    incomplete = copy(A)
    rank_i = rank(incomplete)
    #computes complete matrix rank
    complete = hstack((A, b))
    rank_c = rank(complete)
    
    #compares ranks
    solution = 0
    if abs(rank_i == rank_c):
        is_consistent = True
        print('The linear system is consistent.')
        solution = n - rank_i
        if solution == 0:
            print('It does admit only one solution.')
        else:
            print('It does admit Inf^', solution, 'solution(s):')
            print('they are infinite and depend on', solution, 'parameter(s).')
    else:
        print('The linear system in inconsistent.')
        print('It does not admit any solution.')
        print('Incomplete matrix has rank: ', rank_i, '\nComplete matrix has rank: ', rank_c)  
          
    return is_consistent, solution
        
    

           

#TEST UPPER TRIANGULAR LINEAR SYSTEM
# A = array([[1, -2, 3], [0, -10, 13], [0, 0, 1]])
# b = array([1, 6, 2])
# x = [-1, 2, 2]

#TEST LOWER TRIANGULAR LINEAR SYSTEM
# A = array([[1, 0, 0], [-10, 13, 0], [1, -2, 3]])
# b = array([2, 6, 1])
# x = [2, 2, 1]

#TEST SQUARE MATRICES
# A = array([[1, -2, 3], [4, -10, 13], [5, 1, 1]]) det = 17
# B = array([[1, 1, 0], [2, 1, 1], [3, 0, 1]]) det = 2
# C = array([[1, 0, 1], [-2, -3, 1], [3, 3, 0]]) det = 0
# D = array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
# E = array([[-1,2,-1,2,3], [0,1,0,1,1], [-2,2,-2,1,3]], dtype=float)

#INCONSISTENT SYSTEM
#A = array([[1,1,-2], [1,-1,1], [2,0,-1]])
#b = array([[0,1,0]])

#CONSISTENT SYSTEM
#A = array([[1,1,-2], [1,-1,1], [2,0,-1]])
#b = array([[0,1,1]])

#SQUARE_LINEAR_SYSTEM
#A = array([[1, 1, 0], [2, 1, 1], [3, 0, 1]], dtype=float)
#b = array([[1,1,1]], dtype=float)
#x* = [0.5, 0.5, -0.5]
#A = array([[-1, 0, -2], [2,1,2], [-1,0,-2], [2,1,1], [3,1,3]], dtype=float)
#b = array([[1,1,1,1,1]], dtype=float)
#x* = [0, 9/5, -2/5]