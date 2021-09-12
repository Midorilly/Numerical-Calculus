# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 14:52:37 2021

@author: angel
"""

from numpy import transpose, linalg, set_printoptions, full, copy, array, zeros, asarray, vstack, shape, ones, dot, arange, empty
from pylab import legend, xlabel, ylabel, title, plot, xlim, grid
from linear_algebra import transposition, rank, echelon_form, pivot_index, laplace, gauss_elimination

set_printoptions(precision=4)

def linear_independent(*v):
    """
    OK
    Determines which vectors are linearly independent.
    The algorithm inserts every column-like vector in an emtpy matrix, then proceeds to reduce this
    matrix to echelon form; indexes of pivot columns are searched and stored in a list by pivot_column()

    Parameters
    ----------
    *v : array
        Arbitraty number of vectors. 

    Returns
    -------
    independent : array
        Array of linear independent vectors.
    """  
    vectors = copy(v)
    c = len(vectors)
    r = len(vectors[0])  
        
    M = full((c, r), vectors)
    M = transposition(M)
    
    E = copy(M)  
    E = echelon_form(E)
    pivot_row, pivot_column = pivot_index(E)
       
    independent = []
    for p in pivot_column:
        independent.append(M[:,p])

    print('Column index(es) of linear independet vectors: ', pivot_column)
    return independent

def generator(*v):
    """
    Determines whether the input vectors are a generating set.

    Parameters
    ----------
    *v : array
        Arbitraty number of vectors. 

    Returns
    -------
    generators : array
        Array of generating set vectors.

    """
    vectors = copy(v)
    c = len(vectors)
    r = len(vectors[0])    
    
    M = full((c, r), vectors)
    M = transposition(M)
    
    E = copy(M)  
    E = echelon_form(E)
    pivot_row, pivot_column = pivot_index(E)
       
    generators = []
    for p in pivot_row:
        generators.append(M[p,:])
        
    print('Row index(es) of generator vectors: ', pivot_row)
    return generators
    
def base(*v):
    """
    Determines whether the input vectors are base of a vector space.

    Parameters
    ----------
    *v : array
        Arbitraty number of vectors. 

    Returns
    -------
    is_base: bool
    base: bidimensional array
        Base.

    """
    
    tol=1e-10
    vectors = copy(v)
    c = len(vectors)
    r = len(vectors[0])  
    is_base = False
    
    base = full((c, r), vectors)
    base = transposition(base)
    
    if abs(c-r)<tol and laplace(base)!=0:
        is_base = True
        return is_base, base
    else:
        return is_base
   
def norm(A, s):
    """
    Computes norm 1 or Inf of a generic matrix m-by-n.

    Parameters
    ----------
    A : bidimensional array
        m-by-n matrix.
    s : string
        type of norm.

    Returns
    -------
    norm : floar
        computed norm.

    """
    
    [m,n] = shape(A)
    values = []
    
    if s == '1':
        for j in range(0, n):
            sum = 0
            for i in range(0, m):
                sum = abs(A[i,j]) + sum
            values.append(sum)
        
    elif s == 'inf':
        for i in range(0, m):
            sum = 0
            for j in range(0, n):
                sum = abs(A[i,j]) + sum
            values.append(sum)
            
    norm = max(values)
    return norm
        
def hilbert_matrix(n):
    """
    Generates a n-by-n Hilbert matrix

    Parameters
    ----------
    n : int
        Dimension of the matrix.

    Returns
    -------
    H : bidimensional array
        Hilber matrix.

    """
    x = int(n)
    H = empty((x, x))
    for i in range(0, x):
        for j in range(0, x):
                H[i,j] = 1/((i+1)+(j+1)-1)            
    return H
                
def hilbert_system(n):
    """
    Solve a linear system using its theoric solution e, n-by-1 array of ones. 
    Its matrix of coefficients is the Hilbert matrix.

    Parameters
    ----------
    n : int
        Dimension of the Hilbert matrix.

    Returns
    -------
    x : array
        Vector n-by-1 of variables.

    """
    
    H = hilbert_matrix(n)
    e = ones((n))
    b = dot(H, e)
    
    x = linalg.solve(H, b)
    #x = inverse_linear_system(H, b)
    print('x:', x)
    return x

def hilbert_cond(n):
    """
    Computes the conditioning on the solution of a linear system whose matrix of 
    coefficients is an Hilbert matrix.

    Parameters
    ----------
    n : int
        Starting from dimension 1, computes matrices until dimension n is reached.

    Returns
    -------
    None.

    """
    
    y = []
    for i in range(1, n+1):
       H = hilbert_matrix(i)
       b = hilbert_system(i)
       c = linalg.cond(vstack((H, b)))
       print(c)
       y.append(c)
    x = arange(1, n+1, 1)  
    xlim(1, n)
    #ylim(0, 10)
    title('Condizionamento della matrice di Hilbert')
    xlabel('Dimensione n')
    ylabel('CONDIZIONAMENTO della soluzione')
    grid(axis='both')
    plot(x, y, label="Condizionamento")
    legend(loc='upper left')

def hilbert_RE(n):
    """
    Computer relative error on the theoric solution and the actual one of a Hilbert 
    linear system.

    Parameters
    ----------
    n : int
        Starting from dimension 1, computes matrices until dimension n is reached.

    Returns
    -------
    None.

    """
    
    for i in range(1, n+1):
        b = hilbert_system(i)
        e = ones((i))
        y = abs(b - e)/abs(e)
    x = arange(1, n+1, 1)  
    xlim(1, n)
    #ylim(0, 10)
    title('Condizionamento della matrice di Hilbert')
    xlabel('Dimensione n')
    ylabel('ERRORE RELATIVO sulla soluzione')
    grid(axis='both')
    plot(x, y, label="Errore relativo")
    legend(loc='upper left')




#TEST LINEAR INDEPENDENCE
#v1 = array([-1,2,-1,2,3], dtype=float)
#v2 = array([0,1,0,1,1], dtype=float)
#v3 = array([-2,2,-2,1,3], dtype=float)
#v4 = array([1,1,1,1,1], dtype=float)

#v5 = array([-2,0,1,2])
#v6 = array([-1,-1,2,-2])
#v7 = array([2,2,-2,-2])

#v1 = array([-1, 1, -1])
#v2 = array([-3, 3, -3])
#v3 = array([2, 1, 5])
#v4 = array([-3, 0, -6])
#pivot = [0,2]

#BASE
#e1 = array([1,0,0])
#e2 = array([0,1,0])
#e3 = array([0,0,1])
