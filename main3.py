import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itds_module as itds


def titolo(t: str):
    print("\n" + "=" * 80)
    print(t)
    print("=" * 80)


def sotto(t: str):
    print("\n" + "-" * 80)
    print(t)
    print("-" * 80)


def load_dataset_from_uci():
    """
    Dataset UCI Wine:
    - prima colonna: classe (1..3)
    - altre colonne: feature continue
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    cols = [
        "class",
        "alcohol","malic_acid","ash","alcalinity_of_ash","magnesium",
        "total_phenols","flavanoids","nonflavanoid_phenols","proanthocyanins",
        "color_intensity","hue","od280/od315_of_diluted_wines","proline"
    ]
    df = pd.read_csv(url, header=None, names=cols)
    y = df["class"].to_numpy()  # classi vere (ground truth)
    X = df.drop(columns=["class"]).to_numpy(dtype=float)
    feature_names = df.drop(columns=["class"]).columns.to_list()
    return X, y, feature_names


def main():
    # =========================================================================
    # PUNTO 1) Implementazione kmeans(D,G) che ritorna D1..DG
    # =========================================================================
    titolo("PUNTO 1) Implementazione kmeans(D, G) -> restituisce D1..DG")

    print("Nel file itds_module.py è presente la funzione kmeans(D, G).")
    print("La funzione NON usa sklearn KMeans o funzioni già pronte.")
    print("Output della funzione:")
    print("- clusters = [D1, D2, ..., DG] (sub-matrici)")
    print("- labels   = cluster assegnato a ciascun punto (utile per TCE e plot)")
    print("- centroids, n_iter, objective (informazioni utili per i test)\n")

    # =========================================================================
    # PUNTO 2) Scelta dataset UCI con feature continue
    # =========================================================================
    titolo("PUNTO 2) Scelta dataset UCI (feature continue)")

    X, y, feat = load_dataset_from_uci()
    n, d = X.shape
    print("Dataset scelto: Wine (UCI)")
    print(f"Campioni = {n}")
    print(f"Feature  = {d}")
    print("Esempio feature:", feat[:5], "...\n")

    # Scelta G (qui coerente con le 3 classi del dataset)
    G = len(np.unique(y))
    print(f"Scelgo G = {G} cluster (coerente con le {G} classi del dataset).\n")

    # Parametri comuni esperimenti
    max_iter = 100
    n_init = 10
    seed = None  # se voglio riproducibilità posso mettere seed=123

    print("Parametri usati per i test:")
    print(f"- max_iter = {max_iter}")
    print(f"- n_init   = {n_init} (ripartenze, tengo la migliore)")
    print(f"- seed     = {seed} (None => non riproducibile)\n")

    # =========================================================================
    # PUNTO 3) Test kmeans + valutazione con Total Cluster Entropy (TCE)
    # =========================================================================
    titolo("PUNTO 3) Test kmeans + valutazione con Total Cluster Entropy (TCE)")

    print("La TCE misura la purezza dei cluster rispetto alle classi vere.")
    print("Più TCE è basso, più i cluster contengono prevalentemente una sola classe.\n")

    # Eseguo una configurazione “base” (kmeans++ + mean)
    clusters, labels, centroids, n_iter, obj = itds.kmeans(
        X, G,
        init="kmeans++",
        centroid_method="mean",
        max_iter=max_iter,
        n_init=n_init,
        seed=seed,
        standardize=True
    )

    print("Risultato kmeans (config base: init=kmeans++, centroid=mean)")
    print("Dimensioni delle matrici D1..DG:")
    for k in range(G):
        print(f"D{k+1}: {clusters[k].shape[0]} x {clusters[k].shape[1]}")
    print(f"Iterazioni: {n_iter}")
    print(f"Objective : {obj:.4f} (SSE perché centroid_method='mean')")

    tce_base = itds.total_cluster_entropy(labels, y, G=G)
    print(f"TCE        : {tce_base:.4f}\n")

    # =========================================================================
    # PUNTO 4) Test strategie inizializzazione
    # PUNTO 5) Test metodi per computare il centroide
    # =========================================================================
    titolo("PUNTO 4-5) Test: strategie inizializzazione (P4) e metodi centroide (P5)")

    configs = [
        ("random",   "mean"),
        ("kmeans++", "mean"),
        ("random",   "median"),
        ("kmeans++", "median"),
    ]

    risultati = []
    for init, cmeth in configs:
        sotto(f"Config: init={init} | centroid_method={cmeth}")

        clusters_c, labels_c, centroids_c, n_iter_c, obj_c = itds.kmeans(
            X, G,
            init=init,
            centroid_method=cmeth,
            max_iter=max_iter,
            n_init=n_init,
            seed=seed,
            standardize=True
        )

        tce_c = itds.total_cluster_entropy(labels_c, y, G=G)

        print(f"Iterazioni: {n_iter_c}")
        print(f"Objective : {obj_c:.4f} (SSE se mean, L1 se median)")
        print(f"TCE       : {tce_c:.4f}")
        sizes = [clusters_c[k].shape[0] for k in range(G)]
        print(f"Dimensione cluster: {sizes}")

        risultati.append((tce_c, init, cmeth, labels_c))

    best_tce, best_init, best_cmeth, best_labels = min(risultati, key=lambda x: x[0])

    titolo("Migliore configurazione (min TCE)")
    print(f"init={best_init}, centroid_method={best_cmeth}, TCE={best_tce:.4f}\n")

    # =========================================================================
    # PUNTO 6) Plot scatter
    # =========================================================================
    titolo("PUNTO 6) Scatter plot (PCA 2D)")

    print("Per lo scatter 2D uso PCA:")
    print("- PCA1 e PCA2 sono le prime due componenti principali.")
    print("- La PCA serve solo per visualizzare, il clustering è fatto sulle 13 feature.\n")

    Z = itds.pca_2d(X)

    # Scatter con colori = cluster trovato
    plt.figure(figsize=(9.5, 6.2))
    plt.scatter(Z[:, 0], Z[:, 1], c=best_labels, s=35, edgecolors="k", linewidths=0.3)
        

    plt.title(f"k-means clustering (PCA 2D)\ninit={best_init}, centroid={best_cmeth} | TCE={best_tce:.4f}")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show()

    # Scatter con colori = classi vere (per confronto)
    plt.figure(figsize=(9.5, 6.2))
    plt.scatter(Z[:, 0], Z[:, 1], c=y, s=35, edgecolors="k", linewidths=0.3)
   

    plt.title("Wine dataset (PCA 2D) - Colori = classi vere")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show()

    titolo("FINE ASSIGNMENT 3")
    print("Checklist punti:")
    print("1) kmeans(D,G) implementata e restituisce D1..DG")
    print("2) dataset UCI scelto")
    print("3) valutazione con TCE")
    print("4) test strategie inizializzazione")
    print("5) test metodo centroide")
    print("6) scatter plot")


if __name__ == "__main__":
    main()
