import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Wczytanie danych
dane = pd.read_csv("../Lab2/Lab_2-zad_4.csv", delimiter=";")

# Ilość sąsiadów
n_neighbors = 3

# Model 1 - Zależność kalorii względem tłuszczy zawartych w produkcie
dane["tluszcz"] = dane["tluszcz"] > 1
dane["tluszcz"] = dane["tluszcz"].astype("category").cat.codes
x1 = np.array(dane["kalorie"]).reshape(-1, 1)
y1 = np.array(dane["tluszcz"])
model1 = KNeighborsClassifier(n_neighbors)
model1.fit(x1, y1)
print(f"Dopasowanie modelu 1: {model1.score(x1, y1)}")
# Wypisanie przewidywań dla wartości kalorii w zakresie 0-200 z krokiem 1
print(f"Przewidywania modelu 1:\n{model1.predict(np.arange(0, 200).reshape(-1, 1))}")

# Model 2 - Zależność ilości cukru od występowania na półce nr 1
x2 = np.array(dane["cukry"]).reshape(-1, 1)
y2 = np.array(dane["polka_1"])
model2 = KNeighborsClassifier(n_neighbors)
model2.fit(x2, y2)
print(f"Dopasowanie modelu 2: {model2.score(x2, y2)}")
# Wypisanie przewidywań dla wartości cukru w zakresie 0-20 z krokiem 1
print(f"Przewidywania modelu 2:\n{model2.predict(np.arange(0, 20).reshape(-1, 1))}")

# Model 3 - Zależność występowania cukru w składzie w zależności od ilości potasu
dane["potas"] = dane["potas"] > 180
dane["potas"] = dane["potas"].astype("category").cat.codes
x3 = np.array(dane["cukry"]).reshape(-1, 1)
y3 = np.array(dane["potas"])
model3 = KNeighborsClassifier(n_neighbors)
model3.fit(x3, y3)
print(f"Dopasowanie modelu 3: {model3.score(x3, y3)}")
# Wypisanie przewidywań dla wartości cukru w zakresie 0-20 z krokiem 1
print(f"Przewidywania modelu 3:\n{model3.predict(np.arange(0, 20).reshape(-1, 1))}")
