import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Wczytanie danych
dane = pd.read_csv("lab_2-zad_4.csv", delimiter=";")

# Model 1: Regresja logistyczna: zależność kalorii od tłuszczu
dane["tluszcz"] = dane["tluszcz"] > 0
print(dane["tluszcz"])
dane["tluszcz"] = dane["tluszcz"].astype("category")
print(dane["tluszcz"])
dane["tluszcz"] = dane["tluszcz"].cat.codes
print(dane["tluszcz"])

# TODO: Stworzenie modelu logitowego i wyświetlenie wykresu
