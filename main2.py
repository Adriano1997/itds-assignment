"""
ITDS - main2.py (TEST COMPLETI ASSIGNMENT)
=========================================

Questo script implementa TUTTI i punti richiesti dalla traccia:

1) Selezione dataset con feature continue e classe discreta:
   - Wine dataset (sklearn): 178 campioni, 13 feature, 3 classi

2) Classificatore Bayes con stimatore pdf multivariato:
   - Gaussiano multivariato per classe (covarianza piena)

3) Naive Bayes con stimatore pdf univariato:
   - Istogrammi per feature e per classe (parametri: bins, alpha smoothing)

4) Gaussian Naive Bayes:
   - Feature indipendenti Gaussiane univariate per classe (parametro: reg_eps)

5) Accuratezza media (split 50/50 per classe) su più run

6) Confronto accuratezze medie variando parametri degli stimatori pdf
   - Bayes MVN: reg_eps
   - Gaussian NB: reg_eps
   - NB Istogramma: bins e alpha

In più (utile e richiesto nella pratica):
- Confusion matrix per ciascun classificatore (sul singolo split)
- Grafici 2D stile richiesto: colore=classe reale, marker=classe predetta
  (esempio: alcohol vs malic_acid)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import itds_module as itds


# ----------------------------
# Utility di stampa (terminal)
# ----------------------------

def stampa_titolo(t: str):
    print("\n" + "=" * 80)
    print(t)
    print("=" * 80)


def stampa_sottotitolo(t: str):
    print("\n" + "-" * 80)
    print(t)
    print("-" * 80)


def print_confusion(cm: np.ndarray, class_names):
    """
    Stampa confusion matrix leggibile:
    righe = classe reale, colonne = classe predetta
    """
    print("Confusion matrix (righe=reale, colonne=predetta):")
    header = " " * 12 + " ".join([f"{c:>10}" for c in class_names])
    print(header)
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>12} " + " ".join([f"{v:10d}" for v in row]))


# ----------------------------
# Dataset
# ----------------------------

def load_wine_dataset():
    """
    Carica Wine dataset da sklearn.
    """
    from sklearn.datasets import load_wine
    data = load_wine()
    X = data["data"].astype(float)
    y = data["target"]
    class_names = data["target_names"]
    feature_names = data["feature_names"]
    return X, y, class_names, feature_names


# ----------------------------
# Main
# ----------------------------

def main():
    # 1) DATASET
    stampa_titolo("PUNTO 1) DATASET")
    X, y, class_names, feature_names = load_wine_dataset()

    print(f"Dataset: {X.shape[0]} campioni, {X.shape[1]} feature")
    print(f"Classi: {list(class_names)}")
    print("Esempio feature:", feature_names[:5], "...")
    print()

    # Split 50/50 per classe (richiesto dalla traccia)
    Xtr, ytr, Xte, yte = itds.stratified_split_half_per_class(X, y, seed=7)
    print(f"Split 50/50 per classe:")
    print(f"  Training set: {Xtr.shape[0]} campioni")
    print(f"  Test set    : {Xte.shape[0]} campioni")

    # Feature per grafico 2D stile screenshot
    feat_x = "alcohol"
    feat_y = "malic_acid"
    if feat_x not in feature_names or feat_y not in feature_names:
        # fallback se per qualche motivo i nomi non matchano
        feat_x = feature_names[-1]
        feat_y = feature_names[1]
        print(f"[WARN] Feature richieste non trovate. Uso fallback: {feat_x} vs {feat_y}")

    # ---------------------------------------------------------------------
    # PUNTO 2) BAYES MULTIVARIATO
    # ---------------------------------------------------------------------
    stampa_titolo("PUNTO 2) BAYES GAUSSIANO MULTIVARIATO (full-cov)")

    mvn_model, mvn_prep = itds.train_bayes_gaussian_multivariate(
        Xtr, ytr,
        reg_eps=1e-6,              # regolarizzazione covarianza (stabilità numerica)
        use_standardization=True,  # standardizzo: migliora conditioning delle cov
    )
    yhat_mvn = itds.predict_bayes_gaussian_multivariate(mvn_model, Xte, mvn_prep)
    acc_mvn = itds.accuracy(yte, yhat_mvn)

    print(f"Accuratezza (singolo split): {acc_mvn*100:.2f}%")
    cm_mvn = itds.confusion_matrix(yte, yhat_mvn, labels=np.unique(y))
    print_confusion(cm_mvn, class_names)

    stampa_sottotitolo("GRAFICO 2D (colore=reale, marker=predetta) - Bayes MVN")
    itds.plot_risultati_2d(
        Xte, yte, yhat_mvn,
        feature_names=feature_names,
        class_names=class_names,
        feat_x=feat_x,
        feat_y=feat_y,
        title="Risultati Classificazione (2D) - Bayes Multivariato"
    )
    plt.show()

    # ---------------------------------------------------------------------
    # PUNTO 3) NAIVE BAYES ISTOGRAMMA (pdf univariata)
    # ---------------------------------------------------------------------
    stampa_titolo("PUNTO 3) NAIVE BAYES (ISTOGRAMMA UNIVARIATO)")

    nb_bins = 10
    nb_alpha = 1.0

    nb_model, nb_prep = itds.train_naive_bayes_histogram(
        Xtr, ytr,
        n_bins=nb_bins,            # parametro stimatore pdf
        alpha=nb_alpha,            # smoothing (evita prob=0)
        use_standardization=True,  # fondamentale: binning dipende dalla scala
    )
    yhat_nb = itds.predict_naive_bayes_histogram(nb_model, Xte, nb_prep)
    acc_nb = itds.accuracy(yte, yhat_nb)

    print(f"Parametri: bins={nb_bins}, alpha={nb_alpha}")
    print(f"Accuratezza (singolo split): {acc_nb*100:.2f}%")
    cm_nb = itds.confusion_matrix(yte, yhat_nb, labels=np.unique(y))
    print_confusion(cm_nb, class_names)

    stampa_sottotitolo("GRAFICO 2D (colore=reale, marker=predetta) - NB Istogramma")
    itds.plot_risultati_2d(
        Xte, yte, yhat_nb,
        feature_names=feature_names,
        class_names=class_names,
        feat_x=feat_x,
        feat_y=feat_y,
        title=f"Risultati Classificazione (2D) - NB Istogramma (bins={nb_bins})"
    )
    plt.show()

    # ---------------------------------------------------------------------
    # PUNTO 4) GAUSSIAN NAIVE BAYES
    # ---------------------------------------------------------------------
    stampa_titolo("PUNTO 4) GAUSSIAN NAIVE BAYES")

    gnb_eps = 1e-9
    gnb_model, gnb_prep = itds.train_gaussian_naive_bayes(
        Xtr, ytr,
        reg_eps=gnb_eps,           # evita varianza zero
        use_standardization=False  # qui può stare anche False (Wine già ben scalato)
    )
    yhat_gnb = itds.predict_gaussian_naive_bayes(gnb_model, Xte, gnb_prep)
    acc_gnb = itds.accuracy(yte, yhat_gnb)

    print(f"Parametro: reg_eps={gnb_eps}")
    print(f"Accuratezza (singolo split): {acc_gnb*100:.2f}%")
    cm_gnb = itds.confusion_matrix(yte, yhat_gnb, labels=np.unique(y))
    print_confusion(cm_gnb, class_names)

    stampa_sottotitolo("GRAFICO 2D (colore=reale, marker=predetta) - Gaussian NB")
    itds.plot_risultati_2d(
        Xte, yte, yhat_gnb,
        feature_names=feature_names,
        class_names=class_names,
        feat_x=feat_x,
        feat_y=feat_y,
        title="Risultati Classificazione (2D) - Gaussian NB"
    )
    plt.show()

    # ---------------------------------------------------------------------
    # PUNTO 5) ACCURATEZZA MEDIA SU PIÙ RUN (split 50/50 per classe)
    # ---------------------------------------------------------------------
    stampa_titolo("PUNTO 5) ACCURATEZZE MEDIE SU PIÙ RUN (split 50/50 per classe)")

    runs = 5
    seed0 = 123

    mvn_avg = itds.average_accuracy_over_runs(
        X, y, runs=runs, method="bayes_mvn", seed=seed0,
        reg_eps=1e-6, use_standardization=True
    )
    gnb_avg = itds.average_accuracy_over_runs(
        X, y, runs=runs, method="nb_gauss", seed=seed0,
        reg_eps=1e-9, use_standardization=False
    )
    nb_avg = itds.average_accuracy_over_runs(
        X, y, runs=runs, method="nb_hist", seed=seed0,
        n_bins=10, alpha=1.0, use_standardization=True
    )

    print(f"Numero run: {runs}")
    print(f"Bayes MVN (reg_eps=1e-6)      : {mvn_avg*100:.2f}%")
    print(f"Gaussian NB (reg_eps=1e-9)    : {gnb_avg*100:.2f}%")
    print(f"NB Istogramma (bins=10,a=1.0) : {nb_avg*100:.2f}%")

    # ---------------------------------------------------------------------
    # PUNTO 6) CONFRONTO ACCURATEZZE MEDIE VARIANDO PARAMETRI pdf estimator
    # ---------------------------------------------------------------------
    stampa_titolo("PUNTO 6) COMPARAZIONE MEDIE VARIANDO PARAMETRI DEGLI STIMATORI PDF")

    # 6A) Bayes MVN: reg_eps
    stampa_sottotitolo("6A) Bayes MVN: variazione reg_eps (regolarizzazione covarianza)")
    mvn_eps_list = [1e-9, 1e-6, 1e-3]
    mvn_avg_map = {}
    for eps in mvn_eps_list:
        mvn_avg_map[eps] = itds.average_accuracy_over_runs(
            X, y, runs=runs, method="bayes_mvn", seed=seed0,
            reg_eps=eps, use_standardization=True
        )
        print(f"reg_eps={eps:>8g} -> accuracy media: {mvn_avg_map[eps]*100:.2f}%")

    # 6B) Gaussian NB: reg_eps
    stampa_sottotitolo("6B) Gaussian NB: variazione reg_eps (stabilizza varianze)")
    gnb_eps_list = [1e-12, 1e-9, 1e-6]
    gnb_avg_map = {}
    for eps in gnb_eps_list:
        gnb_avg_map[eps] = itds.average_accuracy_over_runs(
            X, y, runs=runs, method="nb_gauss", seed=seed0,
            reg_eps=eps, use_standardization=False
        )
        print(f"reg_eps={eps:>8g} -> accuracy media: {gnb_avg_map[eps]*100:.2f}%")

    # 6C) NB Istogramma: bins e alpha
    stampa_sottotitolo("6C) NB Istogramma: variazione bins e alpha (smoothing)")
    bins_list = [5, 10, 20]
    alpha_list = [0.1, 1.0]
    nb_hist_map = {}

    for b in bins_list:
        for a in alpha_list:
            nb_hist_map[(b, a)] = itds.average_accuracy_over_runs(
                X, y, runs=runs, method="nb_hist", seed=seed0,
                n_bins=b, alpha=a, use_standardization=True
            )
            print(f"bins={b:2d}, alpha={a:<3} -> accuracy media: {nb_hist_map[(b,a)]*100:.2f}%")

    # Mini-riassunto finale (tabellina)
    stampa_titolo("RIASSUNTO FINALE (PUNTI 2-6)")

    print("Singolo split (seed=7):")
    print(f"  Bayes MVN            : {acc_mvn*100:.2f}%")
    print(f"  NB Istogramma        : {acc_nb*100:.2f}%")
    print(f"  Gaussian NB          : {acc_gnb*100:.2f}%\n")

    print(f"Medie su {runs} run (seed base={seed0}):")
    print(f"  Bayes MVN (eps=1e-6) : {mvn_avg*100:.2f}%")
    print(f"  Gaussian NB (1e-9)   : {gnb_avg*100:.2f}%")
    print(f"  NB Hist (10,1.0)     : {nb_avg*100:.2f}%")

    # Grafico a barre (solo per visualizzare le medie "baseline")
    labels = ["Bayes MVN", "Gaussian NB", "NB Hist"]
    values = [mvn_avg, gnb_avg, nb_avg]

    plt.figure(figsize=(9.5, 4.8))
    plt.bar(labels, values)
    plt.ylabel("Accuracy media")
    plt.ylim(0, 1.0)
    plt.title(f"Accuratezze medie su {runs} run (split 50/50 per classe)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
