import numpy as np
import math
from itds_module import entropy   # o dal nome del file che hai scelto
from itds_module import joint_entropy
from itds_module import conditional_entropy
from itds_module import mutual_information
from itds_module import KL_divergence_discrete
from itds_module import KL_divergence_continuous
import matplotlib.pyplot as plt

# ================== PARTI DISCRETE ==================

p = np.array([0.9, 0.1])
H = entropy(p)
print("Entropy(p) =", H, "bit")

pmf = np.array([0.45, 0.17, 0.21])
H = entropy(pmf)
print("Entropy(p) =", H, "bit")

j_pdf = np.array([
    [0.15, 0.35],
    [0.10, 0.40],
])

Hxy = joint_entropy(j_pdf)
print("Joint Entropy =", Hxy, "bits")

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