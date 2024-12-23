import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression

# Wczytanie danych
dane = pd.read_csv("Lab_2-zad_3.csv", delimiter=";")
X = dane[["Nr miesiaca"]]
Y = dane[["Sumaryczne bledy"]]
X = X.values.reshape(-1, 1)

liniowy = LinearRegression()
liniowy.fit(X, Y)
print(f"M. liniowy: {liniowy.score(X, Y)}")

plt.figure(figsize=(10, 6), dpi=300)
plt.scatter(X, Y, color="green", alpha=0.4)
plt.plot(X, liniowy.predict(X), color="red", label="Model liniowy", linewidth=1)
plt.xlabel("Numer miesiąca")
plt.ylabel("Sumaryczne błędy")
plt.title("Sumaryczne błędy w okresie eksploatacji")
plt.legend()
plt.show()
