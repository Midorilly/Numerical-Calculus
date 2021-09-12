# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 23:44:04 2021

@author: angel
"""
from numpy import sin, cos, exp

def bisezione(f, a, b, tol=1e-10, itmax=100):
    """
    Bisection method
    
    Parameters
    ----------
    f : function
        Funzione di cui cercare lo zero
    a, b : float
        Intervallo in cui calcolare lo zero di f
    tol : float
        Precisione richiesta. The default is 1e-10.
    itmax : int
        Numero massimo di iterate consentite. The default is 100.

    Returns
    -------
    c : float
        Approssimazione di uno zero di f(x).
    it: int
        Numero di iterate eseguite.
    """
    fa = f(a)
    fb = f(b)
    if fa*fb>0:
        print('La funzione non cambia segno agli estremi dell\'intervallo')
        return
    arresto = False
    it = 0 #contatore di iterate
    while not(arresto) and it<itmax:
        it = it+1
        c = (a+b)/2
        fc = f(c)
        if fc == 0:
            return c, it
        elif fa*fc<0:
            b = c
        else:
            a = c
            fa = fc
        arresto = abs(b-a)<tol
    if not(arresto):
        print('Attenzione: precisione non raggiunta')
    return c, it

def direzione_costante(f, m, x0, tol=1e-10, itmax=100):
    """
    Metodo della direzione costante
    
    Parameters
    ----------
    f : function
        Funzione di cui calcolare lo zero.
    m : float
        Parametro iniziale.
    x0 : float
        Stima iniziale dello zero di f.
   tol : float
        Precisione richiesta. The default is 1e-10.
    itmax : int
        Numero assimo di iterate consentite. The default is 100.

    Returns
    -------
    x1: float
        Approssimazione di uno zero di f(x).
    it: int
        Numero di iterate eseguite.
    """
    arresto = False
    it = 0 #contatore di iterate
    while not(arresto) and it<itmax:
        it = it+1
        x1=x0-m*f(x0)
        print(x1)
        arresto = abs(x1-x0)<tol
        x0 = x1
    if not(arresto):
        print('Attenzione: precisione non raggiunta')
    return x1, it

def newton(f, x0, tol=1e-10, itmax=100):
    """
    Metodo di Newton
    
    Parameters
    ----------
    f : function
        Funzione di cui calcolare lo zero.
    x0 : float
        Stima iniziale dello zero di f.
    tol : float
        Precisione richiesta. The default is 1e-10.
    itmax : int
        Numero assimo di iterate consentite. The default is 100.

    Returns
    -------
    x1: float
        Approssimazione di uno zero di f(x). 
    it: int
        Numero di iterate eseguite.
    """
    arresto = False
    it = 0 #contatore di iterate
    while not(arresto) and it<itmax:
        it = it+1
        x1 = x0-f(x0)/f(x0, 1)
        print(x1)
        arresto = abs(x1-x0)<tol
        x0 = x1
    if not(arresto):
         print('Attenzione: precisione non raggiunta')
    return x1, it

def newton_modificato(f, x0, tol=1e-10, itmax=100):
    """
    Metodo di Newton modificato
    
    Parameters
    ----------
    f : function
        Funzione di cui calcolare lo zero.
    x0 : float
        Stima iniziale dello zero di f.
   tol : float
        Precisione richiesta. The default is 1e-10.
    itmax : int
        Numero assimo di iterate consentite. The default is 100.

    Returns
    -------
    x1: float
        Approssimazione di uno zero di f(x). 
    it: int
        Numero di iterate eseguite.
    """
    arresto = False
    it = 0 #contatore di iterate
    m=1/f(x0,1) #parametro prefissato
    while not(arresto) and it<itmax:
        it = it+1
        x1 = x0-f(x0)*m
        print(x1)
        arresto = abs(x1-x0)<tol
        x0 = x1
    if not(arresto):
         print('Attenzione: precisione non raggiunta')
    return x1, it

def secanti(f, x0, x1, tol=1e-10, itmax=100):
    """
    Metodo delle secanti
    
    Parameters
    ----------
   f : function
        Funzione di cui calcolare lo zero.
    x0 : float
        Stima iniziale dello zero di f.
    x1 : float
        Seconda stima iniziale dello zero di f.
   tol : float
        Precisione richiesta. The default is 1e-10.
    itmax : int
        Numero assimo di iterate consentite. The default is 100.

    Returns
    -------
    x2 : float
        Approssimazione di uno zero di f(x). 
    it: int
        Numero di iterate eseguite.
    """ 
    arresto = abs(x1-x0)<tol
    it = 0 #contatore di iterate
    while not(arresto) and it<itmax:
        it = it+1 
        d = (f(x1)-f(x0))/(x1-x0) #denominatore
        x2 = x1 - f(x1)/d
        print(x2)
        arresto = abs(x2-x1)<tol
        x0 = x1
        x1 = x2
    if not(arresto):
        print('Attenzione: precisione non raggiunta')
    return x2, it


"""
FUNZIONI TEST
"""
def f(x,ord=0):
    if ord==0:
        y=x-cos(x)
    elif ord==1:
        y=1+sin(x)
    else:
        print('ordine di derivazione non definito')
        return
    return y
    
def g(x,ord=0):
    if ord==0:
        y=x-exp(-x)
    elif ord==1:
        y=1+exp(-x)
    else:
        print('ordine di derivazione non definito')
        return    
    return y

def h(x,ord=0):
    if ord==0:
        y=x-sin(x) # in zero ha una radice tripla
    elif ord==1:
        y=1-cos(x)
    else:
        print('ordine di derivazione non definito')
        return    
    return y