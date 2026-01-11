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


#!/usr/bin/env python3
# Author: Adriano Marini
# Modulo di supporto per l'Assignment 2 (classificatore di Bayes)

import numpy as np
import pandas as pd


def load_banknote_dataset(from_url=True, local_path="data_banknote_authentication.txt"):
    """
    Carica il dataset "Banknote Authentication" dell'UCI.

    Ogni riga rappresenta una banconota:
        - 4 feature reali (estratte da immagini / wavelet)
        - 1 etichetta discreta (0 = autentica, 1 = falsa)

    Parametri
    ---------
    from_url : bool
        Se True scarica direttamente dal sito UCI.
        Se False legge il file locale 'local_path'.
    local_path : str
        Percorso del file locale (se from_url=False).

    Ritorna
    -------
    X : ndarray di shape (n_samples, 4)
        Matrice delle feature continue.
    y : ndarray di shape (n_samples,)
        Vettore delle etichette (0/1).
    """
    feature_names = ["variance", "skewness", "curtosis", "entropy"]
    columns = feature_names + ["class"]

    if from_url:
        # URL ufficiale del dataset UCI (formato CSV senza intestazione)
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "00267/data_banknote_authentication.txt"
        )
        df = pd.read_csv(url, header=None, names=columns)
    else:
        # Variante offline: leggo il file se già scaricato
        df = pd.read_csv(local_path, header=None, names=columns)

    # Estraggo solo le 4 feature continue e la colonna classe
    X = df[feature_names].values.astype(float)
    y = df["class"].values.astype(int)

    return X, y


def fit_gaussian_bayes(X_train, y_train):
    """
    Stima i parametri del classificatore di Bayes con modello
    gaussiano multivariato per p(x | c).

    Idea:
    -----
    Per ogni classe c:
        - stimiamo la media mu_c (vettore R^d)
        - stimiamo la matrice di covarianza Sigma_c (d x d)
        - stimiamo la prior p(c) come frequenza relativa nel training set

    Tutto viene impacchettato nel dizionario 'params', che poi
    viene usato dalla funzione di predizione.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    n_samples, n_features = X_train.shape
    classes = np.unique(y_train)

    # Dizionario che contiene TUTTO quello che serve al classificatore
    params = {
        "classes": classes,
        "mean": {},        # mu_c
        "inv_cov": {},     # Sigma_c^{-1}
        "log_det_cov": {}, # log |Sigma_c|
        "log_prior": {},   # log p(c)
    }

    for c in classes:
        # Prendo solo le righe del training appartenenti alla classe c
        Xc = X_train[y_train == c]

        # p(c) stimata come frequenza: numero campioni in c / totale
        prior = Xc.shape[0] / n_samples

        # Media vettoriale sulle righe
        mu_c = Xc.mean(axis=0)

        # Covarianza d x d (rowvar=False => righe = osservazioni)
        Sigma_c = np.cov(Xc, rowvar=False)

        # Leggera regolarizzazione diagonale per evitare problemi
        # di inversione se Sigma è quasi singolare
        eps = 1e-6
        Sigma_c = Sigma_c + eps * np.eye(n_features)

        # Pre-calcolo di inverse e log-determinante: così la predizione
        # su molti punti è più veloce (non ricalcolo tutto ogni volta)
        inv_Sigma_c = np.linalg.inv(Sigma_c)
        sign, logdet = np.linalg.slogdet(Sigma_c)

        params["mean"][c] = mu_c
        params["inv_cov"][c] = inv_Sigma_c
        params["log_det_cov"][c] = logdet
        params["log_prior"][c] = np.log(prior)

    return params


def _log_gaussian_pdf(x, mu, inv_cov, log_det_cov):
    """
    Valuta la log-pdf di una gaussiana multivariata N(mu, Sigma) in x.

    Formula:
        log p(x) = -1/2 [ d log(2π) + log |Sigma| + (x - mu)^T Sigma^{-1} (x - mu) ]

    Lavoriamo direttamente in log per evitare problemi numerici
    quando p(x) è molto piccolo.
    """
    x = np.asarray(x)
    d = x.shape[-1]

    diff = x - mu                       # x - mu
    quad = diff.T @ inv_cov @ diff      # forma quadratica

    return -0.5 * (d * np.log(2.0 * np.pi) + log_det_cov + quad)


def bayes_predict_gaussian(X_test, params):
    """
    Classificatore di Bayes con pdf gaussiana multivariata.

    Per ogni punto x nel test:
        - calcola, per ogni classe c,
              score_c = log p(c) + log p(x | c)
        - sceglie la classe con score massimo.

    Parametri
    ---------
    X_test : ndarray (n_test, d)
        Campioni da classificare.
    params : dict
        Parametri stimati da 'fit_gaussian_bayes'.

    Ritorna
    -------
    y_pred : ndarray (n_test,)
        Etichette predette per il test set.
    """
    X_test = np.asarray(X_test)
    n_samples = X_test.shape[0]
    classes = params["classes"]

    y_pred = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        x = X_test[i]
        scores = []  # lista temporanea degli score per ogni classe

        for c in classes:
            mu_c = params["mean"][c]
            inv_cov_c = params["inv_cov"][c]
            log_det_c = params["log_det_cov"][c]
            log_prior_c = params["log_prior"][c]

            # log p(x | c) tramite la gaussiana multivariata
            log_lik = _log_gaussian_pdf(x, mu_c, inv_cov_c, log_det_c)

            # log p(c) + log p(x | c) = log p(c, x) (a costante additiva)
            scores.append(log_prior_c + log_lik)

        # Prendo la classe con score massimo
        best_class = classes[np.argmax(scores)]
        y_pred[i] = best_class

    return y_pred
