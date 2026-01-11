#!/usr/bin/env python3
# Script principale per l'Assignment 2
#
# Qui NON definiamo funzioni "di teoria", ma solo:
# - caricamento del dataset
# - suddivisione train/test
# - training del classificatore di Bayes
# - valutazione su test (accuratezza + matrice di confusione)

import numpy as np
from itds_module import (
    load_banknote_dataset,
    fit_gaussian_bayes,
    bayes_predict_gaussian,
)


def train_test_split_simple(X, y, test_ratio=0.3, random_state=0):
    """
    Piccola funzione "artigianale" per fare train/test split.

    Non usa scikit-learn, così il codice resta completamente "manuale".
    L'idea è:
        - genero una permutazione casuale degli indici
        - prendo la prima parte come test, il resto come train
    """
    X = np.asarray(X)
    y = np.asarray(y)

    rng = np.random.RandomState(random_state)
    n = X.shape[0]

    # Creo un vettore [0, 1, 2, ..., n-1] e lo mescolo
    idx = np.arange(n)
    rng.shuffle(idx)

    # Numero di campioni da mettere nel test set
    n_test = int(round(test_ratio * n))

    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    return X_train, X_test, y_train, y_test


def main():
    # ------------------------------------------------------------------
    # 1) Caricamento del dataset scelto (Banknote Authentication)
    # ------------------------------------------------------------------
    X, y = load_banknote_dataset(from_url=True)
    print(f"Dataset caricato: X shape = {X.shape}, y shape = {y.shape}")

    # ------------------------------------------------------------------
    # 2) Suddivisione in training set e test set
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split_simple(
        X, y, test_ratio=0.3, random_state=42
    )
    print(f"Train: {X_train.shape[0]} campioni  |  Test: {X_test.shape[0]} campioni")

    # ------------------------------------------------------------------
    # 3) Stima dei parametri del classificatore di Bayes
    # ------------------------------------------------------------------
    params = fit_gaussian_bayes(X_train, y_train)
    print("Parametri del modello bayesiano stimati (medie, covarianze, prior).")

    # ------------------------------------------------------------------
    # 4) Predizione delle etichette sul test set
    # ------------------------------------------------------------------
    y_pred = bayes_predict_gaussian(X_test, params)

    # ------------------------------------------------------------------
    # 5) Valutazione: accuratezza e matrice di confusione
    # ------------------------------------------------------------------
    accuracy = np.mean(y_pred == y_test)
    print(f"\nAccuratezza sul test set: {accuracy * 100:.2f}%")

    classes = np.unique(y_test)
    n_classes = len(classes)

    # Costruiamo una matrice C (n_classes x n_classes)
    # C[i, j] = quante volte la vera classe i è stata predetta come j
    C = np.zeros((n_classes, n_classes), dtype=int)
    mapping = {c: i for i, c in enumerate(classes)}

    for yt, yp in zip(y_test, y_pred):
        C[mapping[yt], mapping[yp]] += 1

    print("\nMatrice di confusione (righe = vera classe, colonne = classe predetta):")
    print(C)


if __name__ == "__main__":
    # Eseguo il main solo se lancio direttamente questo file.
    main()
