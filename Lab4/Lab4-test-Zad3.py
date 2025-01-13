import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# # Wczytanie danych z pliku Excel
# dane = pd.read_excel("Płatki-sniadaniowe-cereals.xlsx")  # Załaduj dane z pliku Excel
#
# # Tworzenie klasy binarnej dla potasu (1 = potas > 180, 0 = potas <= 180)
# dane["potas"] = dane["potas"] > 180
# dane["potas"] = dane["potas"].astype("category").cat.codes
#
# # Przygotowanie danych wejściowych i wyjściowych
# x3 = np.array(dane["cukry"]).reshape(-1, 1)  # Ilość cukrów jako cecha wejściowa
# y3 = np.array(dane["potas"])                # Klasa binarna dla potasu
#
# # Normalizacja danych wejściowych
# scaler = MinMaxScaler()
# x3_normalized = scaler.fit_transform(x3)
#
# # Trenowanie modelu KNN
# k = 3
# knn = KNeighborsClassifier(n_neighbors=k)
# knn.fit(x3_normalized, y3)
#
# # Siatka punktów do wizualizacji granicy decyzyjnej
# x_grid = np.linspace(0, 1, 100).reshape(-1, 1)
# y_grid_pred = knn.predict(x_grid)
#
# print(f"Dopasowanie modelu: {knn.score(x3, y3)}")
# print(f"Przewidywania modelu:\n{knn.predict(np.arange(0, 20).reshape(-1, 1))}")
#
# # Wizualizacja
# plt.figure(figsize=(8, 6))
# plt.scatter(x3_normalized, y3, color='blue', edgecolor='k', label='Dane')
# plt.plot(x_grid, y_grid_pred, color='red', label='Granica decyzyjna')
# plt.title(f"Zależność występowania cukru od ilości potasu (k = {k})")
# plt.xlabel("Cukry (znormalizowane)")
# plt.ylabel("Potas > 180 (0 = nie, 1 = tak)")
# plt.legend()
# plt.grid()
# plt.show()

# Wczytanie danych z pliku Excel
dane = pd.read_excel("Płatki-sniadaniowe-cereals.xlsx")  # Załaduj dane z pliku Excel

# Tworzenie klasy binarnej dla potasu (1 = potas > 180, 0 = potas <= 180)
dane["potas"] = dane["potas"] > 180
dane["potas"] = dane["potas"].astype("category").cat.codes

# Przygotowanie danych wejściowych i wyjściowych
x3 = np.array(dane["cukry"]).reshape(-1, 1)  # Ilość cukrów jako cecha wejściowa
y3 = np.array(dane["potas"])                # Klasa binarna dla potasu

# Trenowanie modelu KNN bez normalizacji
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x3, y3)

# Siatka punktów do wizualizacji granicy decyzyjnej
x_grid = np.linspace(min(x3), max(x3), 100).reshape(-1, 1)  # Zmieniamy zakres na rzeczywisty zakres cukrów
y_grid_pred = knn.predict(x_grid)

print(f"Dopasowanie modelu: {knn.score(x3, y3)}")
print(f"Przewidywania modelu:\n{knn.predict(np.arange(0, 20).reshape(-1, 1))}")

# Wizualizacja
plt.figure(figsize=(8, 6))
plt.scatter(x3, y3, color='blue', edgecolor='k', label='Dane')
plt.plot(x_grid, y_grid_pred, color='red', label='Granica decyzyjna')
plt.title(f"Zależność występowania cukru od ilości potasu (k = {k})")
plt.xlabel("Cukry")
plt.ylabel("Potas > 180 (0 = nie, 1 = tak)")
plt.legend()
plt.grid()
plt.show()
