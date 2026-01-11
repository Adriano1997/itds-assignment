#!/usr/bin/env python3
# Author:  Adriano Marini 
# Assignment 1 - FUNCTIONS
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
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

"""
ITDS - Assignment: Bayes / Naive Bayes classifiers
==================================================

Contiene TUTTO ciò che serve per l'assignment:
- Bayes classifier con stimatore pdf multivariato (Gaussiano full-cov)
- Naive Bayes con stimatore pdf univariato (istogramma per feature)
- Gaussian Naive Bayes (Gaussiano univariato indipendente per feature)
- Split stratificato 50/50 per classe (come richiesto)
- Accuracy, confusion matrix
- Plot 2D (PCA) + decision regions
- Accuratezza media su più run e comparazioni parametriche
"""




# ----------------------------
# Helpers
# ----------------------------

def set_seed(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[Sequence] = None) -> np.ndarray:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)

    idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm


def stratified_split_half_per_class(
    X: np.ndarray,
    y: np.ndarray,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split 50/50 PER CLASSE: metà campioni di ogni classe in train, metà in test.
    """
    rng = set_seed(seed)
    X = np.asarray(X)
    y = np.asarray(y)

    classes = np.unique(y)
    train_idx: List[int] = []
    test_idx: List[int] = []

    for c in classes:
        idx = np.where(y == c)[0]
        idx = rng.permutation(idx)
        n_train = len(idx) // 2
        train_idx.extend(idx[:n_train].tolist())
        test_idx.extend(idx[n_train:].tolist())

    train_idx = rng.permutation(np.array(train_idx, dtype=int))
    test_idx = rng.permutation(np.array(test_idx, dtype=int))

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=0)
    sigma = np.where(sigma == 0.0, 1.0, sigma)
    return mu, sigma


def standardize_transform(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (X - mu) / sigma


# ----------------------------
# 1) Bayes Gaussian Multivariate (full covariance)
# ----------------------------

@dataclass
class BayesGaussianMultivariateModel:
    classes_: np.ndarray
    priors_: np.ndarray
    means_: np.ndarray
    covs_: np.ndarray
    inv_covs_: np.ndarray
    log_det_covs_: np.ndarray
    reg_eps_: float


def _logpdf_mvn(X: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray, log_det_cov: float) -> np.ndarray:
    Xc = X - mean
    q = np.einsum("...i,ij,...j->...", Xc, inv_cov, Xc)
    d = X.shape[1]
    return -0.5 * (d * np.log(2.0 * np.pi) + log_det_cov + q)


def train_bayes_gaussian_multivariate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    reg_eps: float = 1e-6,
    use_standardization: bool = True,
) -> Tuple[BayesGaussianMultivariateModel, Dict[str, np.ndarray]]:
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)

    preprocess: Dict[str, np.ndarray] = {}
    X_proc = X_train

    if use_standardization:
        mu, sigma = standardize_fit(X_train)
        preprocess["mu"] = mu
        preprocess["sigma"] = sigma
        X_proc = standardize_transform(X_train, mu, sigma)

    classes = np.unique(y_train)
    K = len(classes)
    n, d = X_proc.shape

    priors = np.zeros(K, dtype=float)
    means = np.zeros((K, d), dtype=float)
    covs = np.zeros((K, d, d), dtype=float)
    inv_covs = np.zeros((K, d, d), dtype=float)
    log_det_covs = np.zeros(K, dtype=float)

    for k, c in enumerate(classes):
        Xc = X_proc[y_train == c]
        priors[k] = len(Xc) / n
        means[k] = np.mean(Xc, axis=0)

        cov = np.cov(Xc, rowvar=False, ddof=0)
        cov = np.asarray(cov, dtype=float) + reg_eps * np.eye(d)

        covs[k] = cov
        inv_covs[k] = np.linalg.inv(cov)

        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise ValueError("Covariance not PD even after regularization.")
        log_det_covs[k] = logdet

    model = BayesGaussianMultivariateModel(
        classes_=classes,
        priors_=priors,
        means_=means,
        covs_=covs,
        inv_covs_=inv_covs,
        log_det_covs_=log_det_covs,
        reg_eps_=reg_eps,
    )
    return model, preprocess


def predict_bayes_gaussian_multivariate(
    model: BayesGaussianMultivariateModel,
    X: np.ndarray,
    preprocess: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    X_proc = X

    if preprocess is not None and "mu" in preprocess and "sigma" in preprocess:
        X_proc = standardize_transform(X_proc, preprocess["mu"], preprocess["sigma"])

    K = len(model.classes_)
    log_post = np.zeros((X_proc.shape[0], K), dtype=float)

    for k in range(K):
        log_like = _logpdf_mvn(X_proc, model.means_[k], model.inv_covs_[k], model.log_det_covs_[k])
        log_prior = np.log(model.priors_[k] + 1e-300)
        log_post[:, k] = log_like + log_prior

    return model.classes_[np.argmax(log_post, axis=1)]


# ----------------------------
# 2) Naive Bayes Histogram (univariate pdf estimator)
# ----------------------------

@dataclass
class NaiveBayesHistogramModel:
    classes_: np.ndarray
    priors_: np.ndarray
    edges_: List[List[np.ndarray]]
    log_probs_: List[List[np.ndarray]]
    n_bins_: int
    alpha_: float


def _fit_hist_1d(x: np.ndarray, n_bins: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmin == xmax:
        xmin -= 1e-6
        xmax += 1e-6
    return np.linspace(xmin, xmax, n_bins + 1)


def _hist_log_probs(x: np.ndarray, edges: np.ndarray, alpha: float) -> np.ndarray:
    counts, _ = np.histogram(x, bins=edges)
    counts = counts.astype(float)
    probs = (counts + alpha) / (np.sum(counts) + alpha * len(counts))
    return np.log(probs + 1e-300)


def train_naive_bayes_histogram(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_bins: int = 10,
    alpha: float = 1.0,
    use_standardization: bool = True,
) -> Tuple[NaiveBayesHistogramModel, Dict[str, np.ndarray]]:
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)

    preprocess: Dict[str, np.ndarray] = {}
    X_proc = X_train
    if use_standardization:
        mu, sigma = standardize_fit(X_train)
        preprocess["mu"] = mu
        preprocess["sigma"] = sigma
        X_proc = standardize_transform(X_train, mu, sigma)

    classes = np.unique(y_train)
    K = len(classes)
    n, d = X_proc.shape

    priors = np.zeros(K, dtype=float)
    edges: List[List[np.ndarray]] = [[None for _ in range(d)] for _ in range(K)]  # type: ignore
    log_probs: List[List[np.ndarray]] = [[None for _ in range(d)] for _ in range(K)]  # type: ignore

    for k, c in enumerate(classes):
        Xc = X_proc[y_train == c]
        priors[k] = len(Xc) / n
        for j in range(d):
            e = _fit_hist_1d(Xc[:, j], n_bins)
            edges[k][j] = e
            log_probs[k][j] = _hist_log_probs(Xc[:, j], e, alpha=alpha)

    model = NaiveBayesHistogramModel(
        classes_=classes,
        priors_=priors,
        edges_=edges,
        log_probs_=log_probs,
        n_bins_=n_bins,
        alpha_=alpha,
    )
    return model, preprocess


def _bin_index(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.digitize(x, edges) - 1
    return np.clip(idx, 0, len(edges) - 2)


def predict_naive_bayes_histogram(
    model: NaiveBayesHistogramModel,
    X: np.ndarray,
    preprocess: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    X_proc = X
    if preprocess is not None and "mu" in preprocess and "sigma" in preprocess:
        X_proc = standardize_transform(X_proc, preprocess["mu"], preprocess["sigma"])

    n, d = X_proc.shape
    K = len(model.classes_)

    log_post = np.zeros((n, K), dtype=float)
    for k in range(K):
        lp = np.full(n, np.log(model.priors_[k] + 1e-300), dtype=float)
        for j in range(d):
            edges = model.edges_[k][j]
            logp_bins = model.log_probs_[k][j]
            idx = _bin_index(X_proc[:, j], edges)
            lp += logp_bins[idx]
        log_post[:, k] = lp

    return model.classes_[np.argmax(log_post, axis=1)]


# ----------------------------
# 3) Gaussian Naive Bayes
# ----------------------------

@dataclass
class GaussianNaiveBayesModel:
    classes_: np.ndarray
    priors_: np.ndarray
    means_: np.ndarray
    vars_: np.ndarray
    reg_eps_: float


def train_gaussian_naive_bayes(
    X_train: np.ndarray,
    y_train: np.ndarray,
    reg_eps: float = 1e-9,
    use_standardization: bool = False,
) -> Tuple[GaussianNaiveBayesModel, Dict[str, np.ndarray]]:
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)

    preprocess: Dict[str, np.ndarray] = {}
    X_proc = X_train
    if use_standardization:
        mu, sigma = standardize_fit(X_train)
        preprocess["mu"] = mu
        preprocess["sigma"] = sigma
        X_proc = standardize_transform(X_train, mu, sigma)

    classes = np.unique(y_train)
    K = len(classes)
    n, d = X_proc.shape

    priors = np.zeros(K, dtype=float)
    means = np.zeros((K, d), dtype=float)
    vars_ = np.zeros((K, d), dtype=float)

    for k, c in enumerate(classes):
        Xc = X_proc[y_train == c]
        priors[k] = len(Xc) / n
        means[k] = np.mean(Xc, axis=0)
        vars_[k] = np.var(Xc, axis=0, ddof=0) + reg_eps

    return GaussianNaiveBayesModel(classes_=classes, priors_=priors, means_=means, vars_=vars_, reg_eps_=reg_eps), preprocess


def predict_gaussian_naive_bayes(
    model: GaussianNaiveBayesModel,
    X: np.ndarray,
    preprocess: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    X_proc = X
    if preprocess is not None and "mu" in preprocess and "sigma" in preprocess:
        X_proc = standardize_transform(X_proc, preprocess["mu"], preprocess["sigma"])

    n = X_proc.shape[0]
    K = len(model.classes_)

    log_post = np.zeros((n, K), dtype=float)
    for k in range(K):
        var = model.vars_[k]
        mu = model.means_[k]
        log_like = -0.5 * (np.sum(np.log(2.0 * np.pi * var)) + np.sum(((X_proc - mu) ** 2) / var, axis=1))
        log_prior = np.log(model.priors_[k] + 1e-300)
        log_post[:, k] = log_like + log_prior

    return model.classes_[np.argmax(log_post, axis=1)]


# ----------------------------
# 2D visualization (PCA + decision regions)
# ----------------------------

def pca_fit_transform(X: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    X = np.asarray(X, dtype=float)
    mu = np.mean(X, axis=0)
    Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:n_components].T
    Z = Xc @ W
    return Z, {"mu": mu, "W": W}


def pca_transform(X: np.ndarray, pca: Dict[str, np.ndarray]) -> np.ndarray:
    return (np.asarray(X, dtype=float) - pca["mu"]) @ pca["W"]


def plot_2d_decision_regions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    predictor_fn,
    title: str,
    grid_points: int = 300,
    ax=None,
):
    import matplotlib.pyplot as plt

    X_all = np.vstack([X_train, X_test])
    _, pca = pca_fit_transform(X_all, n_components=2)
    Z_train = pca_transform(X_train, pca)
    Z_test = pca_transform(X_test, pca)
    Z_all = pca_transform(X_all, pca)

    z0_min, z0_max = Z_all[:, 0].min(), Z_all[:, 0].max()
    z1_min, z1_max = Z_all[:, 1].min(), Z_all[:, 1].max()
    pad0 = 0.08 * (z0_max - z0_min + 1e-12)
    pad1 = 0.08 * (z1_max - z1_min + 1e-12)

    z0 = np.linspace(z0_min - pad0, z0_max + pad0, grid_points)
    z1 = np.linspace(z1_min - pad1, z1_max + pad1, grid_points)
    ZZ0, ZZ1 = np.meshgrid(z0, z1)
    grid_Z = np.column_stack([ZZ0.ravel(), ZZ1.ravel()])

    W = pca["W"]
    mu = pca["mu"]
    grid_X = grid_Z @ W.T + mu
    y_grid = predictor_fn(grid_X).reshape(ZZ0.shape)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))

    ax.set_title(title)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")

    ax.contourf(ZZ0, ZZ1, y_grid, alpha=0.25)

    ax.scatter(Z_train[:, 0], Z_train[:, 1], c=y_train, marker="o", edgecolors="k",
               linewidths=0.4, s=28, label="train")
    ax.scatter(Z_test[:, 0], Z_test[:, 1], c=y_test, marker="x", edgecolors="k",
               linewidths=0.8, s=40, label="test")
    ax.legend(loc="best")
    return ax


# ----------------------------
# Average accuracy utility
# ----------------------------

def average_accuracy_over_runs(
    X: np.ndarray,
    y: np.ndarray,
    runs: int,
    method: str,
    seed: int = 0,
    **kwargs,
) -> float:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    accs: List[float] = []
    for r in range(runs):
        Xtr, ytr, Xte, yte = stratified_split_half_per_class(X, y, seed=seed + r)

        if method == "bayes_mvn":
            model, prep = train_bayes_gaussian_multivariate(Xtr, ytr, **kwargs)
            yhat = predict_bayes_gaussian_multivariate(model, Xte, prep)
        elif method == "nb_hist":
            model, prep = train_naive_bayes_histogram(Xtr, ytr, **kwargs)
            yhat = predict_naive_bayes_histogram(model, Xte, prep)
        elif method == "nb_gauss":
            model, prep = train_gaussian_naive_bayes(Xtr, ytr, **kwargs)
            yhat = predict_gaussian_naive_bayes(model, Xte, prep)
        else:
            raise ValueError(f"Unknown method: {method}")

        accs.append(accuracy(yte, yhat))

    return float(np.mean(accs))
def plot_risultati_2d(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: Sequence[str],
    class_names: Sequence[str],
    feat_x: str,
    feat_y: str,
    title: str = "Risultati Classificazione (2D)",
    ax=None,
):
    """
    Grafico 2D nello stile richiesto:
    - asse X = una feature scelta (es. alcohol)
    - asse Y = una feature scelta (es. malic_acid)
    - COLORE = classe reale
    - MARKER/FORMA = classe predetta

    Nota: questo NON mostra regioni di decisione, ma solo i punti come nel tuo esempio.
    """

    import matplotlib.pyplot as plt

    X = np.asarray(X, dtype=float)
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Trovo gli indici delle feature scelte (per nome)
    fx = list(feature_names).index(feat_x)
    fy = list(feature_names).index(feat_y)

    x = X[:, fx]
    y = X[:, fy]

    # Stile (simile al tuo screenshot)
    # Colore = classe reale
    colori_reale = {
        0: "purple",   # classe 1 nel plot
        1: "teal",     # classe 2 nel plot
        2: "gold",     # classe 3 nel plot
    }

    # Marker = classe predetta
    marker_pred = {
        0: "o",  # pred 1
        1: "x",  # pred 2
        2: "s",  # pred 3
    }

    if ax is None:
        fig, ax = plt.subplots(figsize=(9.5, 6.2))

    ax.set_title(f"{title}\n{feat_x} vs {feat_y} — Colore = classe reale, forma = classe predetta")
    ax.set_xlabel(feat_x)
    ax.set_ylabel(feat_y)
    ax.grid(True, alpha=0.4)

    # Per avere una legenda pulita, disegno per combinazioni (reale, predetta) presenti
    combinazioni = sorted(set(zip(y_true.tolist(), y_pred.tolist())))

    for (c_reale, c_pred) in combinazioni:
        mask = (y_true == c_reale) & (y_pred == c_pred)

        # Etichetta tipo: "Reale 1, Pred 2"
        lab = f"Reale {c_reale+1}, Pred {c_pred+1}"

        # Attenzione: marker 'x' è “unfilled”, se metti edgecolors può uscire warning.
        mk = marker_pred.get(c_pred, "o")
        col = colori_reale.get(c_reale, "gray")

        if mk == "x":
            ax.scatter(
                x[mask], y[mask],
                c=col, marker=mk,
                s=55, linewidths=1.8,
                label=lab
            )
        else:
            ax.scatter(
                x[mask], y[mask],
                c=col, marker=mk,
                s=55, edgecolors="k", linewidths=0.6,
                label=lab
            )

    ax.legend(loc="best")
    return ax
