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


plt.figure(figsize=(10, 6), dpi=300)
plt.scatter(X, Y, color="green", alpha=0.1)

plt.plot(X, liniowy.predict(X), color="red", label="Model liniowy", linewidth=1)

sns.regplot(x=X, y=Y, logistic=True, ci=None, label="Model logisyczny")

plt.xlabel("Rokstudiów")
plt.ylabel("Stancywilny")
plt.title("Logistyczny model prawdopodobieństwa")
plt.legend()
plt.show()


# macierz(dane)
