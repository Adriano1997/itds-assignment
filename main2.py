import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time

# Importa il tuo modulo personalizzato
import itds_module

# --- 1. Caricamento Dataset (Wine) ---
# Link ufficiale UCI Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"

# Nomi delle colonne specificati nella documentazione del dataset
column_names = [
    "Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
    "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins",
    "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"
]

print("1. Scaricando il Wine Dataset...")
# Caricamento dati
df = pd.read_csv(url, names=column_names, header=None)
print(f"   Dataset caricato: {df.shape[0]} righe, {df.shape[1]} colonne")

# Separazione Target (Classe) e Features
# Nel Wine dataset la classe è la PRIMA colonna (indici 1, 2, 3)
y = df.iloc[:, 0].values 
X = df.iloc[:, 1:].values 

# --- 2. Preparazione Dati ---
# Divisione in Training (70%) e Test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"   Training set: {X_train.shape[0]} campioni")
print(f"   Test set: {X_test.shape[0]} campioni")

# --- 3. Addestramento (Training Phase) ---
print("\n2. Inizio Training...")
classifier = itds_module.GaussianBayesClassifier()

start_train = time.time()
classifier.fit(X_train, y_train) # Chiama la funzione nel modulo itsds
end_train = time.time()

print(f"   Training completato in {end_train - start_train:.5f} secondi.")

# --- 4. Predizione (Test Phase) ---
print("\n3. Inizio Predizione (Test)...")
start_pred = time.time()
y_pred = classifier.predict(X_test) # Chiama la funzione nel modulo itsds
end_pred = time.time()

# Calcolo del tempo medio per singola predizione (per dimostrare l'efficienza)
avg_time = (end_pred - start_pred) / len(X_test)
print(f"   Predizione completata in {end_pred - start_pred:.5f} secondi.")
print(f"   Tempo medio per campione: {avg_time:.6f} sec (Ottimizzato!)")

# --- 5. Valutazione e Metriche ---
accuracy = accuracy_score(y_test, y_pred)
print(f"\n4. Risultati:")
print(f"   Accuratezza: {accuracy*100:.2f}%")

print("\n   Report di Classificazione:")
print(classification_report(y_test, y_pred))

# --- 6. Visualizzazione ---
print("\n5. Generazione Grafici...")

# Grafico A: Matrice di Confusione
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix - Wine Dataset\n(Gaussian Bayes Classifier)')
plt.xlabel('Classe Predetta')
plt.ylabel('Classe Reale')
plt.tight_layout()
plt.savefig('wine_confusion_matrix.png')
print("   -> Salvato 'wine_confusion_matrix.png'")

# Grafico B: Scatterplot (Alcohol vs Malic Acid)
# Mostriamo come il classificatore ha lavorato su due delle feature principali
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=X_test[:, 0], # Alcohol (Colonna 0 di X_test)
    y=X_test[:, 1], # Malic Acid (Colonna 1 di X_test)
    hue=y_test, 
    palette='viridis', 
    style=y_pred,   # La forma del punto indica la predizione (Cerchio=Giusto, X=Sbagliato se diverso)
    s=100
)
plt.title('Risultati Classificazione (Visualizzazione 2D: Alcohol vs Malic Acid)\nColore = Classe Reale, Forma = Classe Predetta')
plt.xlabel(column_names[1]) # Label Alcohol
plt.ylabel(column_names[2]) # Label Malic Acid
plt.legend(title='Classe Reale')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('wine_scatter_results.png')
print("   -> Salvato 'wine_scatter_results.png'")

plt.show()