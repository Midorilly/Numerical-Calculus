# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 01:08:56 2021

@author: angel
"""

from numpy import set_printoptions, transpose, matmul, nan, zeros, copy, identity, triu, shape, delete, array, hstack, vstack, full

set_printoptions(precision=3)

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
    det : float
        Determinant of A.
    """
    [m,n] = shape(A)
    if m == n:
        if n == 1:
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
                x[i,0] = (b[0,i]-sum)/A[i,i]
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
        tol=1e-15
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
    
def echelon_form(M):
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
    
def linear_independence(*vectors):
    """
    Determine which vectors are linear independent.
    The algorithm inserts every column-like vector in an emtpy matrix, then proceeds to reduce this
    matrix to echelon form; indexes of pivot columns are searched and stored in a list by pivot_column()

    Parameters
    ----------
    *vectors : array
        Arbitraty number of vectors. 

    Returns
    -------
    independent : array
        Array of linear independent vectors.
    pivot : array
        Array of linear independent vectors column indexes.
    """  
    c = len(vectors)
    r = len(vectors[0])   
    
    M = full((c, r), vectors)
    M = transposition(M)
    print('Vectors: \n', M)
    
    E = copy(M)  
    E = echelon_form(E)
    pivot = pivot_column(E)
       
    independent = []
    for p in pivot:
        independent.append(M[:,p])
        
    independent = array(independent)
    independent = transposition(independent)
    print('Column index(es) of linear independet vectors: ')
    return pivot, independent
    
def pivot_column(M):
    """
    Parameters
    ----------
    M : bidimensional array
        Matrix in echelon form.

    Returns
    -------
    pivot : array
        Array of pivot columns indexes.
    """
    [m,n] = shape(M)
    pivot = []
    i = 0
    j = 0
    for i in range(m):
        for j in range(n):
            if M[i,j] != 0:
                pivot.append(j)
                break
    return pivot

def linear_least_squares(A, b):
    At = copy(A)
    A = transpose(A)
    b = transpose(b)
    
    R = copy(A)
    [m,n] = shape(R)
    if rank(R) != min(m,n):
        print('Error: the rank of the input matrix is not its maximum \nUnable to solve the problem')
        return
    
    #calcolo AtA e Atb
    AtA = matmul(At,A)
    #print('At*A =\n', AtA) 
    Atb = matmul(At,b)
    #print('At*b =\n', Atb)
    
    #eliminazione di Gauss
    x = gauss_elimination(AtA, Atb)
    return x
    #print(complete)
    
def gauss_elimination(A, b):
    complete = hstack((A, b))
    echelon = echelon_form(complete)
    
    [i,j] = shape(echelon)
    known = array([echelon[:, j-1]], dtype=float)
    coefficient = delete(echelon, j-1, axis=1)
    variable = upper_triangular(coefficient, known)
    return variable

def generating_set(*vectors):
    
    c = len(vectors)
    r = len(vectors[0])   
    
    incomplete = full((c, r), vectors)
    incomplete = transposition(incomplete)
    M = copy(incomplete)
    rank_i = rank(M)
    print(rank_i)
        
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
# B = array([[1, 1, 0], [2, 1, 1], [3, 0, 1]])
# C = array([[1, 0, 1], [-2, -3, 1], [3, 3, 0]])
# D = array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
# E = array([[-1,2,-1,2,3], [0,1,0,1,1], [-2,2,-2,1,3]], dtype=float)

#TEST LINEAR INDEPENDENCE
#v1 = [-1,2,-1,2,3]
#v2 = [0,1,0,1,1]
#v3 = [-2,2,-2,1,3]
#v4 = [1,1,1,1,1]

#v5 = [-2,0,1,2]
#v6 = [-1,-1,2,-2]
#v7 = [2,2,-2,-2]

#v1 = [-1, 1, -1]
#v2 = [-3, 3, -3]
#v3 = [2, 1, 5]
#v4 = [-3, 0, -6]
#pivot = [0,2]