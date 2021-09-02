# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 14:52:37 2021

@author: angel
"""

from numpy import linalg, set_printoptions, sin, full, copy, array, zeros, asarray, vstack, shape, ones, dot, linspace, arange, empty
from pylab import xlabel, ylabel, title, plot, xlim, ylim, grid
from linear_algebra import transposition, rank, echelon_form, pivot_index, laplace, gauss_elimination, inverse_linear_system

set_printoptions(precision=4)

def linear_independence(*vectors):
    """
    Determine which vectors are linearly independent.
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
    
    E = copy(M)  
    E = echelon_form(E)
    pivot_row, pivot_column = pivot_index(E)
       
    independent = []
    for p in pivot_column:
        independent.append(M[:,p])
        
    independent = array(independent)
    independent = transposition(independent)
    print('Column index(es) of linear independet vectors: ')
    return independent

def generating_set(*vectors):
    
    tol=1e-15
    m = len(vectors[0])  
    n = len(vectors)   
    is_gset = False
    
    G = full((n, m), vectors)
    G = transposition(G)
    M = copy(G)
    rank_g = rank(M)
    

    if abs(rank_g - m)<tol and laplace(M) != 0:
        is_gset = True
        
    return is_gset, G
    
def base(*vectors):
    
    is_base = False
    
    pivot, independent = linear_independence(*vectors)
    print('Linear indipendent vectors: \n', independent)
    is_gset, generator = generating_set(*vectors)
    print('Generators set: \n', generator)
    
    if independent.all() == generator.all():
        is_base = True
    return is_base, independent 

"""
#ALTRIMENTI
def base(*vectors):
    #tol=1e-15
    m = len(vectors[0]) #columns
    n = len(vectors) #rows
    is_base = False
    
    M = full((n, m), vectors)
    M = transposition(M)
    B = copy(M)
    print(B)
    det = laplace(M)
    
    if m == n and det != 0:
        is_base = True
        print('Input vectors are base of vectorial space')
        return is_base, B
    
    return is_base
    """
    
def base(independent, generator):
    
    is_base = False
    if independent.all() == generator.all():
        is_base = True

    return is_base

def orthogonal_complement(*vectors):
    
    m = len(vectors[0])  
    n = len(vectors)   
    print(m,n)
    
    M = vstack((vectors))
    print(M)
    A = copy(M)
    b = asarray(zeros((n)))
    print(b)
    x = gauss_elimination(A, b)
    return x

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
    print(H)
    for i in range(0, x):
        for j in range(0, x):
                H[i,j] = 1/((i+1)+(j+1)-1)            
    return H
                
def hilbert_system(n):
    
    H = hilbert_matrix(n)
    e = ones((n))
    b = dot(H, e)
    
    x = inverse_linear_system(H, b)
    print('x:')
    return x

def hilbert_cond(n):
    
    V = hilbert_matrix(n)
    return linalg.cond(V)

def hilbert_plot(n):
       
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
    plot(x, y)
        

#TEST LINEAR INDEPENDENCE
#v1 = array([[-1,2,-1,2,3]], dtype=float)
#v2 = array([[0,1,0,1,1]], dtype=float)
#v3 = array([[-2,2,-2,1,3]], dtype=float)
#v4 = [1,1,1,1,1]

#v5 = [-2,0,1,2]
#v6 = [-1,-1,2,-2]
#v7 = [2,2,-2,-2]

#v1 = [-1, 1, -1]
#v2 = [-3, 3, -3]
#v3 = [2, 1, 5]
#v4 = [-3, 0, -6]
#pivot = [0,2]

#v1 = array([[-1,1,-1]], dtype = float)
#v2 = array([[-3,3,-3]], dtype =float)
#v3 = array([[2,1,5]], dtype = float)
#v4 = array([[-3,0,-6]], dtype = float)