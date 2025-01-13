import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Załaduj dane z pliku Excel
dane = pd.read_excel("Płatki-sniadaniowe-cereals.xlsx")

# Model 2 - Zależność ilości cukru od występowania na półce nr 1
x2 = np.array(dane["cukry"]).reshape(-1, 1)
y2 = np.array(dane["polka_1"])

# Trening modelu KNN
n_neighbors = 3  # Liczba sąsiadów
model2 = KNeighborsClassifier(n_neighbors)
model2.fit(x2, y2)

# Ocena modelu
print(f"Dopasowanie modelu 2: {model2.score(x2, y2)}")

# Wypisanie przewidywań dla wartości cukru w zakresie 0-20 z krokiem 1
print(f"Przewidywania modelu 2:\n{model2.predict(np.arange(0, 20).reshape(-1, 1))}")

# Wizualizacja wyników

# Tworzenie siatki dla granic decyzyjnych
x_min, x_max = x2.min() - 1, x2.max() + 1  # Ustawiamy zakres na podstawie rzeczywistych danych
xx = np.arange(x_min, x_max, 0.01).reshape(-1, 1)

# Predykcja dla każdego punktu siatki
Z = model2.predict(xx)

# Wykres granic decyzyjnych
plt.figure(figsize=(10, 6))
plt.scatter(x2, y2, c=y2, edgecolor='k', cmap=plt.cm.Paired, label="Dane")
plt.plot(xx, Z, color="red", label="Granica decyzyjna")
plt.title(f"Zależność ilości cukru od występowania na półce nr 1 (k = {n_neighbors})")
plt.xlabel("Cukry")
plt.ylabel("Półka nr 1 (0 = nie, 1 = tak)")
plt.legend()
plt.grid(True)
plt.show()
