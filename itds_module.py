#!/usr/bin/env python3
# Author:  Adriano Marini 
# Assignment 1 - FUNCTIONS

import numpy as np
import math
from scipy.integrate import quad


def entropy(p):
    """
    Calcolo dell’entropia di una variabile discreta.
    Prende in ingresso una PMF p = [p1, p2, ..., pN]
    e restituisce H in bit.
    """

    # Converto in array numpy così posso fare calcoli vettoriali
    p = np.array(p, dtype=float)

    # Tolgo gli zeri per evitare problemi con log(0)
    p_nonzero = p[p > 0]

    # Formula di Shannon: - ∑ p log2 p
    H = -np.sum(p_nonzero * np.log2(p_nonzero))

    return float(H)



def joint_entropy(jpdf):
    """
    Entropia congiunta di due variabili discrete.
    La funzione riceve una matrice contenente p(x,y).
    """

    jpdf = np.array(jpdf, dtype=float)

    # Stesso trucco di prima: consideriamo solo valori > 0
    p_nonzero = jpdf[jpdf > 0]

    # Formula H(X,Y) = - ∑ p(x,y) log2 p(x,y)
    Hxy = -np.sum(p_nonzero * np.log2(p_nonzero))

    return float(Hxy)



def conditional_entropy(j_pdf, pY):
    """
    Calcolo dell’entropia condizionata H(X|Y).
    Richiede la matrice congiunta e la marginale di Y.
    
    H(X|Y) = -∑ p(x,y) log2 p(x|y)
    """

    j_pdf = np.array(j_pdf, dtype=float)
    pY = np.array(pY, dtype=float)

    # Dimensioni della matrice congiunta
    X, Y = j_pdf.shape

    H = 0.0

    # scorriamo y (colonne)
    for y in range(Y):
        # se p(y)=0, non contribuisce
        if pY[y] == 0:
            continue

        # scorriamo x (righe)
        for x in range(X):
            p_xy = j_pdf[x, y]

            if p_xy > 0:
                # definizione di probabilità condizionata
                p_x_given_y = p_xy / pY[y]
                H -= p_xy * math.log2(p_x_given_y)

    return H



def mutual_information(j_pdf, pX, pY):
    """
    Informazione mutua I(X;Y)
    I misura quanto sapere Y riduce l'incertezza su X.
    """

    j_pdf = np.array(j_pdf, dtype=float)
    pX = np.array(pX, dtype=float)
    pY = np.array(pY, dtype=float)

    # Costruisco la matrice p(x)p(y)
    px_py = np.outer(pX, pY)

    # Considero solo le entrate dove p(x,y)>0
    mask = j_pdf > 0

    # Formula: ∑ p(x,y) log2( p(x,y) / (p(x)p(y)) )
    I = np.sum(j_pdf[mask] * np.log2(j_pdf[mask] / px_py[mask]))

    return float(I)



def KL_divergence_discrete(p, q):
    """
    Divergenza di Kullback-Leibler tra due PMF discrete.
    Misura quanto Q si discosta da P.
    """
    
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)

    D = 0.0

    for pi, qi in zip(p, q):

        # se pi = 0, quel termine non contribuisce
        if pi == 0:
            continue

        # se qi = 0 mentre pi>0 → divergenza infinita
        if qi == 0:
            return math.inf

        D += pi * math.log2(pi / qi)

    return D



def KL_divergence_continuous(p, q, a=-np.inf, b=np.inf):
    """
    KL divergenza per variabili continue.
    p e q devono essere funzioni (pdf) e a,b sono i limiti di integrazione.
    
    D = ∫ p(x) log( p(x)/q(x) ) dx
    """

    def integrand(x):
        px = p(x)
        qx = q(x)

        # Se p(x)=0, quel punto non contribuisce
        if px == 0:
            return 0.0

        # Se q(x)=0 ma p(x)>0 → divergenza infinita
        if qx == 0:
            return math.inf

        return px * math.log(px / qx, 2)

    # quad integra numericamente la funzione integrand
    result, _ = quad(integrand, a, b)

    return result



from scipy.stats import multivariate_normal

class GaussianBayesClassifier:
    """
    Classificatore Bayesiano basato su stimatore di densità Gaussiana Multivariata.
    Ideale per minimizzare il processing time in fase di test.
    """
    def __init__(self):
        self.model = {}
        self.classes = None

    def fit(self, X, y):
        """
        Fase di Training: stima i parametri della distribuzione per ogni classe.
        
        Args:
            X (numpy array): Matrice delle feature (n_samples, n_features)
            y (numpy array): Vettore delle etichette (n_samples)
        """
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        
        for c in self.classes:
            # Seleziona i dati appartenenti solo alla classe 'c'
            X_c = X[y == c]
            
            # 1. Calcolo del Prior P(omega_i)
            # Probabilità a priori che un dato appartenga alla classe c
            prior = len(X_c) / n_samples
            
            # 2. Stima della Media mu (Mean Vector) - Slide MLE
            mean = np.mean(X_c, axis=0)
            
            # 3. Stima della Covarianza Sigma (Covariance Matrix) - Slide MLE
            # rowvar=False indica che le colonne sono le variabili
            cov = np.cov(X_c, rowvar=False)
            
            # Gestione stabilità numerica: aggiunge un epsilon se la matrice è singolare
            # (opzionale ma consigliato per evitare crash su dataset piccoli)
            cov += np.eye(n_features) * 1e-6
            
            # Salviamo tutto nel modello
            self.model[c] = {
                'prior': prior,
                'mean': mean,
                'cov': cov,
                # Pre-calcoliamo l'oggetto "congelato" per velocizzare la predizione
                'dist': multivariate_normal(mean=mean, cov=cov)
            }
            
    def predict(self, X):
        """
        Fase di Test: predice la classe usando la regola di Bayes.
        Minimizza il processing time perché calcola solo la formula della gaussiana.
        
        Args:
            X (numpy array): Dati di test (n_samples, n_features)
        Returns:
            predictions (numpy array): Classi predette
        """
        predictions = []
        
        for x in X:
            posteriors = []
            
            for c in self.classes:
                prior = self.model[c]['prior']
                dist = self.model[c]['dist']
                
                # Calcolo della Likelihood p(x | omega_i)
                # Formula della densità Gaussiana Multivariata
                likelihood = dist.pdf(x)
                
                # Calcolo della Posteriori (non normalizzata)
                # P(omega_i | x) ∝ p(x | omega_i) * P(omega_i)
                posterior = likelihood * prior
                posteriors.append(posterior)
            
            # Regola di decisione MAP (Maximum A Posteriori)
            # Sceglie la classe con il valore più alto
            predicted_class = self.classes[np.argmax(posteriors)]
            predictions.append(predicted_class)
            
        return np.array(predictions)