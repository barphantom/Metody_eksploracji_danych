import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Wczytanie danych
dane = pd.read_csv("lab2-zad1.csv", delimiter=";")

# Sprawdź krzyżowanie się danych
def macierz(dane):
    plt.figure(figsize=(10, 6), dpi=300)
    scatter_matrix(dane, figsize=(12, 12), diagonal='kde')
    plt.suptitle("Macierz rozrzutu danych")
    plt.tight_layout()
    plt.show()

# Stworzenie liniowego modelu prawdopodobieństwa na podstawie danych oraz utworzenie wykresu
X = dane[["Rokstudiów"]]
Y = dane[["Stancywilny"]]
X = X.values.reshape(-1, 1)
liniowy = LinearRegression()
liniowy.fit(X, Y)
print(f"M. liniowy: {liniowy.score(X, Y)}")
# liniowy.score(X, Y)

X = dane[["Rokstudiów"]]
Y = dane[["Stancywilny"]]
X = X.values.reshape(-1, 1)
logistyczny = LogisticRegression()
logistyczny.fit(X, Y)
print(f"M. logistyczny: {logistyczny.score(X, Y)}")

# Wykres liniowy
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color="grey", marker="x", alpha=0.4)
plt.plot(X, liniowy.predict(X), color="red", label="Model liniowy", linewidth=1)
plt.title("Liniowy model prawdopodobieństwa")
plt.xlabel("Rokstudiów")
plt.ylabel("Stancywilny")
plt.grid(alpha=0.4, linestyle="--", linewidth=0.5, color="gray")
plt.savefig("Lab_2-zad_1-model_lin.png", dpi=300)
plt.show()

# Wykres logistyczny
plt.figure(figsize=(10, 6))
sns.regplot(data=dane, x=X, y=Y, marker="x", color=".3", line_kws=dict(color="r"), logistic=True, ci=20)
plt.title("Logistyczny model prawdopodobieństwa")
plt.xlabel("Rokstudiów")
plt.ylabel("Stancywilny")
plt.grid(alpha=0.4, linestyle="--", linewidth=0.5, color="gray")
plt.savefig("Lab_2-zad_1-model_log.png", dpi=300)
plt.show()