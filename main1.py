import numpy as np
import math
from itds_module import entropy  
from itds_module import joint_entropy
from itds_module import conditional_entropy
from itds_module import mutual_information
from itds_module import KL_divergence_discrete
from itds_module import KL_divergence_continuous
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.datasets import load_iris

import numpy as np



# ================== PARTI DISCRETE ==================
def demo_entropy():
    """
    Esempi semplici di entropia per variabili aleatorie discrete.
    Serve a mostrare come cambia H(X) al variare della PMF.
    """

    print("=== Esempi di ENTROPY ===")

    # Moneta equa: testa/croce con stessa probabilità
    p_fair = np.array([0.5, 0.5])
    H_fair = entropy(p_fair)
    print(f"Moneta equa p={p_fair} -> H(X) = {H_fair:.4f} bit")

    # Moneta sbilanciata: testa molto probabile
    p_biased = np.array([0.9, 0.1])
    H_biased = entropy(p_biased)
    print(f"Moneta sbilanciata p={p_biased} -> H(X) = {H_biased:.4f} bit")

    # Variabile con 3 stati (es: semaforo rosso/giallo/verde)
    p_three = np.array([0.2, 0.5, 0.3])
    H_three = entropy(p_three)
    print(f"Variabile a 3 stati p={p_three} -> H(X) = {H_three:.4f} bit")

    print()  # riga vuota


demo_entropy()
def demo_joint_entropy():
    """
    Esempi semplici per illustrare la Joint Entropy H(X, Y)
    con diversi tipi di dipendenza/indipendenza tra variabili.
    """

    print("\n=== Esempi di JOINT ENTROPY ===")

    # -------------------------
    # Esempio 1: Variabili indipendenti
    # X = {0,1} con p = (0.5, 0.5)
    # Y = {0,1} con p = (0.5, 0.5)
    # Joint = prodotto delle marginali
    # -------------------------
    j1 = np.array([
        [0.25, 0.25],
        [0.25, 0.25]
    ])
    H1 = joint_entropy(j1)
    print(f"1) Variabili indipendenti:\nJoint =\n{j1}\nH(X,Y) = {H1:.4f} bit\n")

    # -------------------------
    # Esempio 2: Variabili debolmente dipendenti
    # Leggera correlazione
    # -------------------------
    j2 = np.array([
        [0.30, 0.20],
        [0.10, 0.40]
    ])
    H2 = joint_entropy(j2)
    print(f"2) Variabili debolmente dipendenti:\nJoint =\n{j2}\nH(X,Y) = {H2:.4f} bit\n")

    # -------------------------
    # Esempio 3: Variabili completamente dipendenti (Y=X)
    # La joint è concentrata sugli eventi X=Y
    # -------------------------
    j3 = np.array([
        [0.5, 0.0],
        [0.0, 0.5]
    ])
    H3 = joint_entropy(j3)
    print(f"3) Variabili completamente dipendenti (Y = X):\nJoint =\n{j3}\nH(X,Y) = {H3:.4f} bit\n")
demo_joint_entropy()
j_pdf = np.array([
    [0.15, 0.35,0.2],
    [0.10, 0.40,0.1],
])

Hxy = joint_entropy(j_pdf)
print("Joint Entropy =", Hxy, "bits")
def demo_conditional_entropy():
    """
    Esempi per verificare il comportamento della Conditional Entropy H(X|Y).
    Utilizza tre joint PMF: indipendenza, dipendenza parziale e dipendenza totale.
    """

    print("\n=== Esempi di CONDITIONAL ENTROPY ===")

    # -------------------------------
    # 1) X e Y indipendenti
    # -------------------------------
    j1 = np.array([
        [0.25, 0.25],
        [0.25, 0.25]
    ])
    pY1 = j1.sum(axis=0) # marginale p(y): somma sulle righe

    H1 = conditional_entropy(j1, pY1)
    print(f"\nCaso 1 - Variabili indipendenti")
    print(f"Joint PMF:\n{j1}")
    print(f"H(X|Y) = {H1:.4f} bit")


    # -------------------------------
    # 2) Dipendenza parziale
    # -------------------------------
    j2 = np.array([
        [0.30, 0.20],
        [0.10, 0.40]
    ])
    pY2 = j2.sum(axis=0)

    H2 = conditional_entropy(j2, pY2)
    print(f"\nCaso 2 - Dipendenza parziale")
    print(f"Joint PMF:\n{j2}")
    print(f"H(X|Y) = {H2:.4f} bit")


    # -------------------------------
    # 3) Dipendenza totale (Y = X)
    # -------------------------------
    j3 = np.array([
        [0.5, 0.0],
        [0.0, 0.5]
    ])
    pY3 = j3.sum(axis=0)

    H3 = conditional_entropy(j3, pY3)
    print(f"\nCaso 3 - Dipendenza totale (Y determina X)")
    print(f"Joint PMF:\n{j3}")
    print(f"H(X|Y) = {H3:.4f} bit")

    print()  # riga vuota finale
demo_conditional_entropy()
def demo_mutual_information():
    """
    Esempi di informazione mutua I(X;Y) per tre casi:
    1) X e Y indipendenti
    2) X e Y moderatamente dipendenti
    3) X e Y completamente dipendenti (X = Y)
    """

    print("\n=== Esempi di MUTUAL INFORMATION ===")

    # -------------------------------
    # Caso 1: variabili indipendenti
    # -------------------------------
    j1 = np.array([
        [0.25, 0.25],
        [0.25, 0.25]
    ])
    pX1 = j1.sum(axis=1)   # marginale di X
    pY1 = j1.sum(axis=0)   # marginale di Y

    I1 = mutual_information(j1, pX1, pY1)

    print("\nCaso 1 - Variabili indipendenti")
    print("Joint PMF:")
    print(j1)
    print(f"p(X) = {pX1}")
    print(f"p(Y) = {pY1}")
    print(f"I(X;Y) = {I1:.4f} bit")
    # Atteso: I(X;Y) = 0 bit (nessuna informazione condivisa)

    # --------------------------------------
    # Caso 2: dipendenza moderata tra X e Y
    # --------------------------------------
    j2 = np.array([
        [0.3, 0.2],
        [0.1, 0.4]
    ])
    pX2 = j2.sum(axis=1)
    pY2 = j2.sum(axis=0)

    I2 = mutual_information(j2, pX2, pY2)

    print("\nCaso 2 - Variabili moderatamente dipendenti")
    print("Joint PMF:")
    print(j2)
    print(f"p(X) = {pX2}")
    print(f"p(Y) = {pY2}")
    print(f"I(X;Y) = {I2:.4f} bit")
    # Atteso: I(X;Y) > 0 ma < 1 bit (dipendenza debole)

    # -------------------------------------------
    # Caso 3: dipendenza totale (X e Y coincidono)
    # -------------------------------------------
    j3 = np.array([
        [0.5, 0.0],
        [0.0, 0.5]
    ])
    pX3 = j3.sum(axis=1)
    pY3 = j3.sum(axis=0)

    I3 = mutual_information(j3, pX3, pY3)

    print("\nCaso 3 - Variabili completamente dipendenti (X = Y)")
    print("Joint PMF:")
    print(j3)
    print(f"p(X) = {pX3}")
    print(f"p(Y) = {pY3}")
    print(f"I(X;Y) = {I3:.4f} bit")
    # Atteso: I(X;Y) = 1 bit (massima informazione condivisa per variabili binarie)

    print()  # riga vuota finale

demo_mutual_information()
def demo_KL_discrete():
    """
    Esempi di divergenza di Kullback-Leibler per PMF discrete.
    Usiamo due coppie di distribuzioni:
    - SET 1: P e Q molto simili
    - SET 2: P e Q molto diversi
    Calcoliamo sia D_KL(P||Q) sia D_KL(Q||P) per mostrare che non è simmetrica.
    """

    print("\n=== Esempi di KL DISCRETA ===")

    # ------------------
    # SET 1: simili
    # ------------------
    P1 = [0.40, 0.35, 0.25]
    Q1 = [0.42, 0.33, 0.25]

    D_P1_Q1 = KL_divergence_discrete(P1, Q1)
    D_Q1_P1 = KL_divergence_discrete(Q1, P1)

    print("\nSET 1 (distribuzioni simili):")
    print(f"P1 = {P1}")
    print(f"Q1 = {Q1}")
    print(f"D_KL(P1 || Q1) = {D_P1_Q1:.6f} bit")
    print(f"D_KL(Q1 || P1) = {D_Q1_P1:.6f} bit")

    # ------------------
    # SET 2: molto diversi
    # ------------------
    P2 = [0.70, 0.20, 0.10]
    Q2 = [0.10, 0.40, 0.50]

    D_P2_Q2 = KL_divergence_discrete(P2, Q2)
    D_Q2_P2 = KL_divergence_discrete(Q2, P2)

    print("\nSET 2 (distribuzioni molto diverse):")
    print(f"P2 = {P2}")
    print(f"Q2 = {Q2}")
    print(f"D_KL(P2 || Q2) = {D_P2_Q2:.6f} bit")
    print(f"D_KL(Q2 || P2) = {D_Q2_P2:.6f} bit")

    print()  # riga vuota finale
demo_KL_discrete()
# marginale p(y): somma sulle righe
pY = j_pdf.sum(axis=0)
H_X_given_Y = conditional_entropy(j_pdf, pY)

print("p(Y)=", pY)
print("H(X|Y) =", H_X_given_Y, "bits")

j_pdf = np.array([
    [0.31, 0.11],
    [0.12, 0.08],
])

# marginali X e Y
pX = j_pdf.sum(axis=1)   # somma per colonne
pY = j_pdf.sum(axis=0)   # somma per righe

print("p(X) =", pX)
print("p(Y) =", pY)

I = mutual_information(j_pdf, pX, pY)
print("Mutual Information I(X;Y) =", I, "bits")

P1 = [0.40, 0.35, 0.25]
Q1 = [0.42, 0.33, 0.25]

P2 = [0.70, 0.20, 0.10]
Q2 = [0.10, 0.40, 0.50]

print("SET 1 (simili):")
print("D_KL(P1 || Q1) =", KL_divergence_discrete(P1, Q1), "bit")
print("D_KL(Q1 || P1) =", KL_divergence_discrete(Q1, P1), "bit")
print()

print("SET 2 (molto diversi):")
print("D_KL(P2 || Q2) =", KL_divergence_discrete(P2, Q2), "bit")
print("D_KL(Q2 || P2) =", KL_divergence_discrete(Q2, P2), "bit")

# ================== PARTI CONTINUE (GAUSSIANE) ==================

def gaussian_pdf(mu, sigma):
    """
    Restituisce la pdf di una Gaussiana N(mu, sigma^2) come funzione p(x).
    """
    coef = 1.0 / (math.sqrt(2 * math.pi) * sigma)

    def pdf(x):
        return coef * math.exp(- (x - mu)**2 / (2 * sigma**2))

    return pdf

# Esempio 1: N(0,1) vs N(0.2, 1.1)
mu1, sigma1 = 0.0, 1.0
mu2, sigma2 = 0.2, 1.1

p = gaussian_pdf(mu1, sigma1)
q = gaussian_pdf(mu2, sigma2)

sigma_max = max(sigma1, sigma2)
a = min(mu1, mu2) - 6 * sigma_max
b = max(mu1, mu2) + 6 * sigma_max

D = KL_divergence_continuous(p, q, a=a, b=b)
print("\nD_KL(N(0,1) || N(0.2,1.1)) =", D, "bit")

# Esempio 2: N(0,1) vs N(1,2)
mu1, sigma1 = 0.0, 1.0
mu2, sigma2 = 1.0, 2.0

p = gaussian_pdf(mu1, sigma1)
q = gaussian_pdf(mu2, sigma2)

sigma_max = max(sigma1, sigma2)
a = min(mu1, mu2) - 6 * sigma_max
b = max(mu1, mu2) + 6 * sigma_max

D = KL_divergence_continuous(p, q, a=a, b=b)
print("D_KL(N(0,1) || N(1,2)) =", D, "bit")


def analyze_vs_mu(mu1=0.0, sigma1=1.0, sigma2=1.0,
                  mu2_min=-3, mu2_max=3, N=40):
    """
    Analizza la Kullback-Leibler divergence D_KL(P||Q)
    quando varia la media della seconda gaussiana (mu2).

    Parametri:
    ----------
    mu1 : float
        Media della prima gaussiana.
    sigma1 : float
        Deviazione standard della prima gaussiana.
    sigma2 : float
        Deviazione standard della seconda gaussiana.
    mu2_min, mu2_max : float
        Estremi dell'intervallo in cui varia mu2.
    N : int
        Numero di punti (risoluzione del grafico).
    """

    print("\n--- ANALISI KL vs differenza tra le medie (mu1 - mu2) ---")

    # ------------------------------------------------------------------
    # Creo N valori equidistanziati tra mu2_min e mu2_max.
    # Questa è proprio la variazione di mu2 che vogliamo studiare.
    #
    # np.linspace(a,b,N) → crea N numeri tra a e b.
    # ------------------------------------------------------------------
    mu2_values = np.linspace(mu2_min, mu2_max, N)

    KL_values = []   # Lista dove salveremo i valori della KL

    # Loop: per ogni mu2 calcolo D_KL(P||Q)
    for mu2 in mu2_values:

        # Costruisco le PDF gaussiane con i parametri attuali
        p = gaussian_pdf(mu1, sigma1)   # P = N(mu1, sigma1^2)
        q = gaussian_pdf(mu2, sigma2)   # Q = N(mu2, sigma2^2)

        # ------------------------------------------------------------------
        # Scelgo un intervallo di integrazione numerica.
        # Le gaussiane sono praticamente zero fuori ±6 σ.
        # ------------------------------------------------------------------
        sigma_max = max(sigma1, sigma2)
        a = min(mu1, mu2) - 6 * sigma_max
        b = max(mu1, mu2) + 6 * sigma_max

        # Calcolo numericamente il KL usando la tua funzione
        KL = KL_divergence_continuous(p, q, a, b)

        # Aggiungo alla lista
        KL_values.append(KL)

    # ------------------------------------------------------------------
    # GRAFICO
    # Asse x: differenza tra le medie (mu1 - mu2)
    # Asse y: valore della KL
    # ------------------------------------------------------------------
    plt.figure()
    plt.plot(mu1 - mu2_values, KL_values, marker="o")
    plt.xlabel(r"$\mu_1 - \mu_2$")
    plt.ylabel(r"$D_{KL}(P||Q)$")
    plt.title("KL divergence vs differenza tra le medie")
    plt.grid(True)

def analyze_vs_sigma(mu1=0.0, mu2=0.0, sigma1=1.0,
                     sigma2_min=0.3, sigma2_max=3.0, N=40):
    """
    Analizza la Kullback-Leibler divergence D_KL(P||Q)
    quando varia la deviazione standard della seconda gaussiana (sigma2).

    Parametri:
    ----------
    mu1, mu2 : float
        Medie delle gaussiane P e Q.
    sigma1 : float
        Deviazione standard della prima gaussiana.
    sigma2_min, sigma2_max : float
        Range entro cui varia sigma2.
    N : int
        Numero di punti del grafico.
    """

    print("\n--- ANALISI KL vs differenza tra le sigma (sigma1 - sigma2) ---")

    # ------------------------------------------------------------------
    # Genero N valori di sigma2 da sigma2_min a sigma2_max.
    # Anche qui linspace crea valori equidistanziati.
    # ------------------------------------------------------------------
    sigma2_values = np.linspace(sigma2_min, sigma2_max, N)

    KL_values = []

    for sigma2 in sigma2_values:

        # PDF delle due gaussiane
        p = gaussian_pdf(mu1, sigma1)
        q = gaussian_pdf(mu2, sigma2)

        # Intervallo di integrazione
        sigma_max = max(sigma1, sigma2)
        a = mu1 - 6 * sigma_max
        b = mu1 + 6 * sigma_max

        # Calcolo della KL
        KL = KL_divergence_continuous(p, q, a, b)
        KL_values.append(KL)

    # ------------------------------------------------------------------
    # GRAFICO
    # Asse x: differenza sigma1 - sigma2
    # ------------------------------------------------------------------
    plt.figure()
    plt.plot(sigma1 - sigma2_values, KL_values, marker="o")
    plt.xlabel(r"$\sigma_1 - \sigma_2$")
    plt.ylabel(r"$D_{KL}(P||Q)$")
    plt.title("KL divergence vs differenza tra le sigma")
    plt.grid(True)


analyze_vs_mu(mu1=0, sigma1=1, sigma2=1, mu2_min=-5, mu2_max=5, N=80)
analyze_vs_sigma(mu1=0, mu2=0, sigma1=1, sigma2_min=0.2, sigma2_max=4, N=80)
plt.show()

def compute_pmf(values):
    """
    Calcola la PMF di un array di numeri interi.
    Restituisce: valori unici + le loro probabilità.
    """
    unique, counts = np.unique(values, return_counts=True)
    pmf = counts / counts.sum()
    return unique, pmf


def plot_pmf(unique, pmf, feature_name):
    """
    Plot della PMF di una feature discreta.
    """
    plt.figure(figsize=(6,4))
    plt.stem(unique, pmf)   
    plt.title(f"PMF della feature: {feature_name}")
    plt.xlabel("Valori discreti")
    plt.ylabel("P(X = x)")
    plt.grid(True)
    plt.show()


def pmf_iris_features():
    """
    Carica il dataset Iris, discretizza le 4 feature
    e plottane la PMF una per una.
    """

    iris = load_iris()
    X = iris.data
    feature_names = iris.feature_names

    # discretizzazione: arrotondo per trasformare i float in interi
    X_discrete = np.round(X).astype(int)

    for i in range(X_discrete.shape[1]):
        feature = X_discrete[:, i]

        # Calcolo PMF
        unique, pmf = compute_pmf(feature)

        # Plot
        plot_pmf(unique, pmf, feature_names[i])


# Chiamata alla funzione (eseguibile)
pmf_iris_features()


def entropy_iris_features():
    """
    Calcola la Shannon entropy delle 4 feature
    del dataset Iris dopo discretizzazione.
    """

    iris = load_iris()
    X = iris.data
    feature_names = iris.feature_names

    # Discretizzazione (come per la PMF)
    X_discrete = np.round(X).astype(int)

    entropies = {}

    for i in range(X_discrete.shape[1]):
        feature = X_discrete[:, i]

        # Ottengo PMF della feature
        unique, counts = np.unique(feature, return_counts=True)
        pmf = counts / counts.sum()

        # Calcolo entropia usando LA TUA funzione
        H = entropy(pmf)

        entropies[feature_names[i]] = H

    return entropies


results = entropy_iris_features()
print("\nEntropy delle feature (dopo discretizzazione):")
for name, H in results.items():
    print(f"{name}: {H:.4f} bit")


def mutual_information_iris():
    """
    Calcola la mutual information tra tutte le coppie
    di feature del dataset Iris (dopo discretizzazione).
    Restituisce la coppia di feature con massima MI.
    """

    iris = load_iris()
    X = iris.data
    feature_names = iris.feature_names

    # Step 1: discretizzazione a interi
    X_discrete = np.round(X).astype(int)

    n_features = X_discrete.shape[1]

    best_pair = None
    best_mi = -1

    # Per salvare tutte le MI se ti servono
    all_mi = {}

    # Step 2: calcolo MI per ogni coppia (i < j)
    for i in range(n_features):
        for j in range(i+1, n_features):

            Xi = X_discrete[:, i]
            Xj = X_discrete[:, j]

            # joint pmf
            joint, counts = np.unique(
                np.column_stack((Xi, Xj)),
                axis=0,
                return_counts=True
            )
            joint_pmf = counts / counts.sum()

            # marginals
            px_vals, px_counts = np.unique(Xi, return_counts=True)
            pX = px_counts / px_counts.sum()

            py_vals, py_counts = np.unique(Xj, return_counts=True)
            pY = py_counts / py_counts.sum()

            # joint pmf in matrice (serve per mutual_information)
            joint_matrix = np.zeros((len(px_vals), len(py_vals)))

            for (xv, yv), p in zip(joint, joint_pmf):
                ix = np.where(px_vals == xv)[0][0]
                iy = np.where(py_vals == yv)[0][0]
                joint_matrix[ix, iy] = p

            # mutual information (usiamo la tua funzione)
            MI = mutual_information(joint_matrix, pX, pY)

            fname = (feature_names[i], feature_names[j])
            all_mi[fname] = MI

            # aggiorno massimo
            if MI > best_mi:
                best_mi = MI
                best_pair = fname

    return best_pair, best_mi, all_mi


pair, value, all_mi = mutual_information_iris()

print("\nMutual information massima:")
print(f"{pair} → {value:.4f} bits")

print("\nTutte le MI:")
for k, v in all_mi.items():
    print(f"{k}: {v:.4f} bits")

def entropy_overall_iris():
    """
    Calcola l'entropia complessiva del dataset Iris
    dopo la discretizzazione delle 4 feature.

    Consideriamo il vettore X = (X1, X2, X3, X4) come
    un'unica variabile aleatoria discreta.
    """

    iris = load_iris()
    X = iris.data                  # shape (150, 4)

    # Discretizzazione (come nelle parti precedenti):
    # arrotondo i float e li trasformo in interi
    X_discrete = np.round(X).astype(int)

    # Ogni riga è una quadrupla (x1, x2, x3, x4).
    # np.unique(axis=0) trova tutte le combinazioni distinte
    unique_rows, counts = np.unique(
        X_discrete, axis=0, return_counts=True
    )

    # Probabilità delle combinazioni (PMF congiunta)
    pmf_joint = counts / counts.sum()

    # Entropia usando la tua funzione entropy(p)
    H_dataset = entropy(pmf_joint)

    print(f"\nEntropy complessiva dell'Iris (dopo discretizzazione): {H_dataset:.4f} bit")

    # Se ti servono anche le combinazioni e le loro probabilità, le ritorni
    return H_dataset, unique_rows, pmf_joint


# Esempio di chiamata:
H_overall, patterns, pmf = entropy_overall_iris()

def plot_mutual_information_heatmap(all_mi):
    """
    Crea una heatmap 4x4 della mutual information tra le feature
    dell'Iris, con etichette ben visibili.
    """

    iris = load_iris()
    feature_names = iris.feature_names

    # Matrice vuota 4x4
    M = np.zeros((4, 4))

    # Inserisco le MI nella matrice (simmetrica)
    for (f1, f2), value in all_mi.items():
        i = feature_names.index(f1)
        j = feature_names.index(f2)
        M[i, j] = value
        M[j, i] = value

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        M,
        annot=True,
        cmap="Blues",
        xticklabels=feature_names,
        yticklabels=feature_names,
        fmt=".2f",
        annot_kws={"size": 12},
        cbar_kws={"shrink": 0.8}
    )

    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.title("Mutual Information tra le feature dell'Iris", fontsize=16)
    plt.tight_layout()
    plt.show()


pair, value, all_mi = mutual_information_iris()
plot_mutual_information_heatmap(all_mi)
