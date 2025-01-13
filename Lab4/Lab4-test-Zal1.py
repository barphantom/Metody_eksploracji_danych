import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Przygotowanie danych (przykład - dostosuj do swojego pliku)
data = pd.read_excel("Płatki-sniadaniowe-cereals.xlsx")  # Załaduj dane z pliku Excel

# Wybieramy cechy (kalorie i tłuszcz) oraz etykiety
X = data[["kalorie", "tluszcz"]].values
y = []

# Tworzenie etykiet klas na podstawie kalorii i tłuszczu
for calories, fat in X:
    if calories <= 110 and fat <= 2:
        y.append(0)  # Niskokaloryczne i niskotłuszczowe
    elif calories > 110 and fat <= 2:
        y.append(1)  # Wysokokaloryczne i niskotłuszczowe
    else:
        y.append(2)  # Wysokokaloryczne i wysokotłuszczowe

# Konwersja etykiet na numpy array
y = np.array(y)

# Normalizacja cech
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Trening modelu KNN
k = 5  # Liczba sąsiadów
knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
knn.fit(X_train, y_train)

# Predykcja na zbiorze testowym
y_pred = knn.predict(X_test)

# Ocena modelu
print("Dokładność:", accuracy_score(y_test, y_pred))
print("Raport klasyfikacji:\n", classification_report(y_test, y_pred, zero_division=0))
print("Przewidywania modelu: ", y_pred)

# Tworzenie siatki dla granic decyzyjnych
x_min, x_max = X_scaled[:, 0].min() - 0.1, X_scaled[:, 0].max() + 0.1
y_min, y_max = X_scaled[:, 1].min() - 0.1, X_scaled[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Predykcja dla każdego punktu siatki
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Wykres granic decyzyjnych
plt.figure(figsize=(10, 6))
# plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')  # Przykład z 'viridis'


# Rozkład danych treningowych i testowych
# Użycie y_pred zamiast y dla danych testowych
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap=plt.cm.Paired, marker='x', s=100, label="Predykcje testowe")

# Rozkład danych treningowych
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, marker='o', label="Dane treningowe")

plt.title(f"Granice decyzyjne KNN (k = {k})")
plt.xlabel("Kalorie (znormalizowane)")
plt.ylabel("Tłuszcz (znormalizowany)")
plt.legend()
plt.grid(True)
plt.show()
