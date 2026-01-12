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
    sigma = np.std(X, axis=0, ddof=1)
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

        cov = np.cov(Xc, rowvar=False, ddof=1)
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
    priors_: np.ndarray                  # shape (K,)
    edges_global_: List[np.ndarray]      # lista di d array di edges (uguali per tutte le classi)
    log_densities_: List[List[np.ndarray]]  # log_densities_[k][j] -> log f bin (len = n_bins)
    n_bins_: int
    alpha_: float


def _fit_hist_edges_global(x: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Crea edges GLOBALI per una feature:
    - stessi edges per tutte le classi (coerente con confronto di pdf per classe)
    """
    x = np.asarray(x, dtype=float)
    xmin = float(np.min(x))
    xmax = float(np.max(x))

    # Evito bin di larghezza zero
    if xmin == xmax:
        xmin -= 1e-6
        xmax += 1e-6

    # Scelgo edges equispaziati (semplice e trasparente)
    edges = np.linspace(xmin, xmax, n_bins + 1)
    return edges


def _hist_log_densities(x: np.ndarray, edges: np.ndarray, alpha: float) -> np.ndarray:
    """
    Stima della DENSITÀ per bin (pdf a tratti costante):
      1) Stimo P(bin | classe) con smoothing (alpha)
      2) Converto in densità: f ≈ P(bin)/width(bin)
      3) Ritorno log densità per stabilità numerica
    """
    counts, _ = np.histogram(x, bins=edges)
    counts = counts.astype(float)

    B = len(edges) - 1                 # numero bin
    widths = np.diff(edges)            # larghezze bin (Δ)

    # Probabilità per bin (con smoothing tipo Laplace)
    probs = (counts + alpha) / (np.sum(counts) + alpha * B)

    # Conversione a densità: f ≈ P(bin)/Δ
    # (così ottengo una vera p.d.f. univariata, come richiesto)
    densities = probs / widths

    return np.log(densities + 1e-300)


def train_naive_bayes_histogram(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_bins: int = 10,
    alpha: float = 1.0,
    use_standardization: bool = True,
) -> Tuple[NaiveBayesHistogramModel, Dict[str, np.ndarray]]:
    """
    Naive Bayes con stimatore univariato a istogrammi.

    VERSIONE "PROF":
    - edges GLOBALI per feature (uguali per tutte le classi)
    - stima di densità per bin: f ≈ P(bin|c)/Δ
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)

    # Pre-processing (opzionale): standardizzazione
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

    # Prior P(c)
    priors = np.zeros(K, dtype=float)

    # Edges globali per ogni feature (d liste)
    edges_global: List[np.ndarray] = []
    for j in range(d):
        edges_global.append(_fit_hist_edges_global(X_proc[:, j], n_bins))

    # log_densities_[k][j] = log densità (pdf) dei bin della feature j per classe k
    log_densities: List[List[np.ndarray]] = [[None for _ in range(d)] for _ in range(K)]  # type: ignore

    for k, c in enumerate(classes):
        Xc = X_proc[y_train == c]
        priors[k] = len(Xc) / n

        for j in range(d):
            edges = edges_global[j]
            log_densities[k][j] = _hist_log_densities(Xc[:, j], edges, alpha=alpha)

    model = NaiveBayesHistogramModel(
        classes_=classes,
        priors_=priors,
        edges_global_=edges_global,
        log_densities_=log_densities,
        n_bins_=n_bins,
        alpha_=alpha,
    )
    return model, preprocess


def _bin_index(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Restituisce l’indice del bin per ogni x:
    - clip ai bordi per valori fuori range
    """
    idx = np.digitize(x, edges) - 1
    return np.clip(idx, 0, len(edges) - 2)


def predict_naive_bayes_histogram(
    model: NaiveBayesHistogramModel,
    X: np.ndarray,
    preprocess: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """
    Predizione Naive Bayes istogramma:
    log P(c|x) ∝ log P(c) + Σ_j log f_j(x_j | c)
    """
    X = np.asarray(X, dtype=float)
    X_proc = X
    if preprocess is not None and "mu" in preprocess and "sigma" in preprocess:
        X_proc = standardize_transform(X_proc, preprocess["mu"], preprocess["sigma"])

    n, d = X_proc.shape
    K = len(model.classes_)

    log_post = np.zeros((n, K), dtype=float)

    for k in range(K):
        # log prior
        lp = np.full(n, np.log(model.priors_[k] + 1e-300), dtype=float)

        # somma dei log delle densità univariate (assunzione naive)
        for j in range(d):
            edges = model.edges_global_[j]
            logdens_bins = model.log_densities_[k][j]
            idx = _bin_index(X_proc[:, j], edges)
            lp += logdens_bins[idx]

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
        vars_[k] = np.var(Xc, axis=0, ddof=1) + reg_eps

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
# =============================================================================
# ASSIGNMENT 
# =============================================================================
import numpy as np

# =============================================================================
# ASSIGNMENT 3 - KMEANS
# =============================================================================
# In questa sezione implemento:
# - PUNTO 1: la funzione kmeans(D, G) che restituisce D1..DG (sub-matrici)
# - PUNTO 3: la Total Cluster Entropy (TCE) per valutare il clustering
# - PUNTO 6: PCA 2D solo per fare lo scatter plot
# =============================================================================


def _standardize_fit(X: np.ndarray):
    """
    Standardizzazione z-score (scelta pratica):
    X_std = (X - mu) / sigma

    La uso perché k-means usa distanze: se le feature hanno scale molto diverse,
    una feature può dominare tutte le altre.
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=0)
    sigma[sigma == 0.0] = 1.0
    return mu, sigma


def _standardize_transform(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return (X - mu) / sigma


def _dist_sqeuclid(X: np.ndarray, C: np.ndarray):
    """
    Distanza euclidea al quadrato tra ogni punto e ogni centroide.
    Restituisce una matrice (n x G):
      dist[i,k] = ||X[i] - C[k]||^2
    """
    n = X.shape[0]
    G = C.shape[0]
    dist = np.zeros((n, G), dtype=float)

    for k in range(G):
        diff = X - C[k]                # (n x d)
        dist[:, k] = np.sum(diff * diff, axis=1)

    return dist


def _dist_l1(X: np.ndarray, C: np.ndarray):
    """
    Distanza Manhattan (L1) tra ogni punto e ogni centroide.
    dist[i,k] = sum_j |X[i,j] - C[k,j]|
    """
    n = X.shape[0]
    G = C.shape[0]
    dist = np.zeros((n, G), dtype=float)

    for k in range(G):
        dist[:, k] = np.sum(np.abs(X - C[k]), axis=1)

    return dist


def _init_random(rng: np.random.Generator, X: np.ndarray, G: int):
    """
    PUNTO 4 (inizializzazione): strategia random
    Scelgo G punti casuali come centroidi iniziali.
    """
    n = X.shape[0]
    idx = rng.choice(n, size=G, replace=False)
    return X[idx].copy()


def _init_kmeanspp(rng: np.random.Generator, X: np.ndarray, G: int):
    """
    PUNTO 4 (inizializzazione): strategia k-means++
    1) primo centroide random
    2) i successivi scelti con probabilità proporzionale alla distanza^2 dal centroide più vicino
    """
    n, d = X.shape
    C = np.zeros((G, d), dtype=float)

    # primo centroide casuale
    C[0] = X[rng.integers(0, n)]

    # distanza^2 dal centroide più vicino (per ogni punto)
    closest_sq = _dist_sqeuclid(X, C[0:1]).ravel()

    for k in range(1, G):
        s = float(np.sum(closest_sq))

        if s <= 1e-12:
            # caso degenerato: punti molto simili
            C[k] = X[rng.integers(0, n)]
        else:
            probs = closest_sq / s
            idx = rng.choice(n, p=probs)
            C[k] = X[idx]

        # aggiorno le distanze minime
        new_sq = _dist_sqeuclid(X, C[k:k+1]).ravel()
        closest_sq = np.minimum(closest_sq, new_sq)

    return C


def _update_centroids_mean(rng: np.random.Generator, X: np.ndarray, labels: np.ndarray, G: int):
    """
    PUNTO 5 (centroide): metodo "mean" -> k-means classico
    Centroide = media dei punti assegnati al cluster.
    """
    d = X.shape[1]
    C = np.zeros((G, d), dtype=float)

    for k in range(G):
        pts = X[labels == k]
        if pts.shape[0] == 0:
            # cluster vuoto: re-inizializzo il centroide con un punto casuale
            C[k] = X[rng.integers(0, X.shape[0])]
        else:
            C[k] = np.mean(pts, axis=0)

    return C


def _update_centroids_median(rng: np.random.Generator, X: np.ndarray, labels: np.ndarray, G: int):
    """
    PUNTO 5 (centroide): metodo "median" -> k-medians
    Centroide = mediana per feature dei punti assegnati al cluster.
    """
    d = X.shape[1]
    C = np.zeros((G, d), dtype=float)

    for k in range(G):
        pts = X[labels == k]
        if pts.shape[0] == 0:
            C[k] = X[rng.integers(0, X.shape[0])]
        else:
            C[k] = np.median(pts, axis=0)

    return C


# =============================================================================
# PUNTO 1) Funzione richiesta: kmeans(D, G) -> restituisce D1..DG
# =============================================================================
def kmeans(
    D: np.ndarray,
    G: int,
    init: str = "kmeans++",            # PUNTO 4: "random" oppure "kmeans++"
    centroid_method: str = "mean",     # PUNTO 5: "mean" oppure "median"
    max_iter: int = 100,
    n_init: int = 10,
    seed=None,
    standardize: bool = True
):
    """
    PUNTO 1 (richiesta traccia):
    Implemento una funzione chiamata kmeans che:
    - prende in input una matrice dati D e un intero G
    - restituisce G matrici D1..DG come risultato del clustering

    Scelte pratiche aggiuntive (per rendere il metodo robusto):
    - init: strategia di inizializzazione (PUNTO 4)
    - centroid_method: come calcolo il centroide (PUNTO 5)
    - n_init: riparto più volte e tengo la migliore (standard per k-means)
    - seed: se lo imposto, rendo riproducibile l’esperimento; se None, ogni run può cambiare
    """

    X = np.asarray(D, dtype=float)
    n, d = X.shape

    if G <= 0:
        raise ValueError("G deve essere >= 1")
    if G > n:
        raise ValueError("G non può essere maggiore del numero di campioni")

    # RNG: seed=None => risultati non riproducibili (ma in media buoni con n_init alto)
    rng_master = np.random.default_rng(seed)

    # Standardizzazione (consigliata)
    if standardize:
        mu, sigma = _standardize_fit(X)
        Xp = _standardize_transform(X, mu, sigma)
    else:
        Xp = X

    best_labels = None
    best_centroids = None
    best_obj = np.inf
    best_iter = 0

    # Riparto n_init volte: tengo la soluzione migliore (objective minima)
    for _ in range(n_init):
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))

        # --- PUNTO 4: inizializzazione centroidi ---
        if init == "random":
            C = _init_random(rng, Xp, G)
        elif init == "kmeans++":
            C = _init_kmeanspp(rng, Xp, G)
        else:
            raise ValueError("init deve essere 'random' o 'kmeans++'")

        labels = np.full(n, -1, dtype=int)

        # --- ciclo k-means: assignment + update ---
        for it in range(1, max_iter + 1):
            # Assignment step
            if centroid_method == "mean":
                dist = _dist_sqeuclid(Xp, C)
            elif centroid_method == "median":
                dist = _dist_l1(Xp, C)
            else:
                raise ValueError("centroid_method deve essere 'mean' o 'median'")

            new_labels = np.argmin(dist, axis=1)

            # Stop se non cambia più nulla
            if it > 1 and np.array_equal(new_labels, labels):
                labels = new_labels
                break

            labels = new_labels

            # Update step (PUNTO 5)
            if centroid_method == "mean":
                C = _update_centroids_mean(rng, Xp, labels, G)
            else:
                C = _update_centroids_median(rng, Xp, labels, G)

        # Calcolo objective per scegliere la migliore run
        if centroid_method == "mean":
            dist = _dist_sqeuclid(Xp, C)
            obj = float(np.sum(dist[np.arange(n), labels]))      # SSE
        else:
            dist = _dist_l1(Xp, C)
            obj = float(np.sum(dist[np.arange(n), labels]))      # L1 cost

        if obj < best_obj:
            best_obj = obj
            best_labels = labels.copy()
            best_centroids = C.copy()
            best_iter = it

    # Costruisco D1..DG nello spazio ORIGINALE (richiesta della traccia)
    clusters = [X[best_labels == k] for k in range(G)]

    # La traccia chiede solo D1..DG; però per i test mi serve anche labels/centroidi.
    # Quindi ritorno una tupla: (clusters, labels, centroids, n_iter, objective).
    return clusters, best_labels, best_centroids, best_iter, best_obj


# =============================================================================
# PUNTO 3) Valutazione con Total Cluster Entropy (TCE)
# =============================================================================
def total_cluster_entropy(cluster_labels: np.ndarray, class_labels: np.ndarray, G: int):
    """
    PUNTO 3:
    Valuto il clustering con la Total Cluster Entropy (TCE):
      TCE = (1/n) * sum_k [ n_k * H_k ]
    dove H_k è l’entropia della distribuzione delle classi vere dentro il cluster k.

    Più TCE è basso => cluster più “puri” rispetto alle classi vere.
    """
    z = np.asarray(cluster_labels).ravel()
    y = np.asarray(class_labels).ravel()

    if z.shape[0] != y.shape[0]:
        raise ValueError("cluster_labels e class_labels devono avere stessa lunghezza")

    n = z.shape[0]
    classes = np.unique(y)
    idx = {c: i for i, c in enumerate(classes)}

    tce = 0.0
    for k in range(G):
        mask = (z == k)
        nk = int(np.sum(mask))
        if nk == 0:
            continue

        counts = np.zeros(len(classes), dtype=float)
        for lab in y[mask]:
            counts[idx[lab]] += 1.0

        # Entropia del cluster k
        s = float(np.sum(counts))
        p = counts / s
        p = p[p > 0]
        Hk = float(-np.sum(p * np.log2(p)))

        tce += nk * Hk

    return float(tce / n)


# =============================================================================
# PUNTO 6) PCA 2D per scatter plot (solo visualizzazione)
# =============================================================================
def pca_2d(X: np.ndarray):
    """
    PUNTO 6:
    PCA 2D per visualizzare dati ad alta dimensione.
    PCA1 e PCA2 sono le prime due componenti principali.
    """
    X = np.asarray(X, dtype=float)
    mu = np.mean(X, axis=0)
    Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:2].T
    Z = Xc @ W
    return Z
