#!/usr/bin/env python3
# Author:  Adriano Marini 
# Assignment 1 - FUNCTIONS

import numpy as np
import math
from scipy.integrate import quad


def entropy(p):
    """
    Calcola la Shannon Entropy di una variabile aleatoria discreta.

    Parameters
    ----------
    p : array-like
        PMF = [p1, p2, ..., pN]

    Returns
    -------
    float
        H(X) in bit.
    """
    # converto in array numpy
    p = np.array(p, dtype=float)

    # rimuovo eventuali zeri per evitare 0 * log2(0)
    p_nonzero = p[p > 0]
    # formula: H = - sum p_i log2 p_i
    H = -np.sum(p_nonzero * np.log2(p_nonzero))
    return float(H)

def joint_entropy(jpdf):
    """
    Compute the joint entropy H(X,Y) from the joint probability mass function.

    Parameters
    ----------
    jpdf : 2D array-like
        Joint PMF p(x,y), e.g.:
        [[p00, p01],
         [p10, p11]]

    Returns
    -------
    float
        Joint entropy in bits.
    """
    jpdf = np.array(jpdf, dtype=float)

    # remove zero entries (to avoid log2(0))
    p_nonzero = jpdf[jpdf > 0]

    Hxy = -np.sum(p_nonzero * np.log2(p_nonzero))
    return float(Hxy)

def conditional_entropy(j_pdf, pY):
    """
    Calcola l'entropia condizionata H(X|Y).

    j_pdf : matrice joint p(x,y)
    pY    : marginale p(y)

    Formula:
       H(X|Y) = - Σx Σy p(x,y) * log2( p(x|y) )
    con:
       p(x|y) = p(x,y) / p(y)
    """

    import numpy as np
    import math

    j_pdf = np.array(j_pdf, dtype=float)
    pY    = np.array(pY, dtype=float)

    X, Y = j_pdf.shape

    H = 0.0  # accumulatore dell'entropia

    # ciclo su Y (colonne)
    for y in range(Y):
        if pY[y] == 0:
            continue  # se p(y)=0 salto

        # ciclo su X (righe)
        for x in range(X):
            p_xy = j_pdf[x, y]
            if p_xy > 0:
                p_x_given_y = p_xy / pY[y]    # p(x|y)
                H -= p_xy * math.log2(p_x_given_y)

    return H


def mutual_information(j_pdf, pX, pY):
    """
    Mutual Information I(X;Y)

    I(X;Y) = Σx Σy p(x,y) log2( p(x,y) / (p(x)p(y)) )

    Parameters
    ----------
    j_pdf : 2D array-like
        Joint PMF p(x,y)
    pX : 1D array-like
        Marginal PMF p(x)
    pY : 1D array-like
        Marginal PMF p(y)

    Returns
    -------
    float
        Mutual Information in bits.
    """

    j_pdf = np.array(j_pdf, dtype=float)
    pX = np.array(pX, dtype=float)
    pY = np.array(pY, dtype=float)

    # matrice p(x)p(y)
    px_py = np.outer(pX, pY)

    # maschera per evitare log(0)
    mask = j_pdf > 0

    I = np.sum(j_pdf[mask] * np.log2(j_pdf[mask] / px_py[mask]))
    return float(I)


def KL_divergence_discrete(p, q):
    """
    Kullback-Leibler divergence D_KL(P || Q)
    per due pmf discrete P e Q.

    Formula:
        D_KL(P || Q) = sum p_i * log2(p_i / q_i)
    """

    import numpy as np
    import math

    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)

    D = 0.0

    for pi, qi in zip(p, q):
        if pi == 0:
            continue            # 0 * log(0/qi) = 0
        if qi == 0:
            return math.inf     # divergenza infinita

        D += pi * math.log2(pi / qi)

    return D


def KL_divergence_continuous(p, q, a=-np.inf, b=np.inf):
    """
    Kullback-Leibler divergence D_KL(P || Q) per pdf continue.

    p, q : funzioni p(x) e q(x)
    a, b : limiti di integrazione (default: -inf, +inf)

    Formula:
        D_KL = ∫ p(x) * log( p(x) / q(x) ) dx
    """

    def integrand(x):
        px = p(x)
        qx = q(x)

        if px == 0:
            return 0.0
        if qx == 0:
            return math.inf   # divergenza infinita

        return px * math.log(px / qx, 2)   # log base 2

    result, _ = quad(integrand, a, b)
    return result
