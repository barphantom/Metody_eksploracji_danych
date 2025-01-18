import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Wczytanie danych
dane = pd.read_csv("dane.txt", delimiter="\t")
print(dane)

# Zmienne do modelu
X = dane[["Density"]]  # Używamy tylko zmiennej density
y = dane["Pct.BF"]

# Dodanie stałej do modelu (dla Statsmodels)
X_with_const = sm.add_constant(X)

# Regresja liniowa (Statsmodels)
model = sm.OLS(y, X_with_const).fit()
print("\nPodsumowanie modelu:")
print(model.summary())

# Regresja liniowa (sklearn)
reg = LinearRegression()
reg.fit(X, y)


# Wyniki regresji
y_pred = reg.predict(X)
print("\nRegresja liniowa (sklearn):")
print("R^2 score:", r2_score(y, y_pred))
print("MSE:", mean_squared_error(y, y_pred))
print("Współczynniki regresji:", reg.coef_)
print("Intercept:", reg.intercept_)

# Wykres rzeczywiste vs przewidywane wartości
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, color='blue', label="Predykcja")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label="Idealna linia")
plt.xlabel("Rzeczywiste wartości Pct.BF")
plt.ylabel("Przewidywane wartości Pct.BF")
plt.title("Rzeczywiste vs Przewidywane wartości Body Fat (Pct.BF)")
plt.legend()
plt.savefig("WykresBF-Density.png", dpi=300, transparent=None, bbox_inches='tight', pad_inches=0.1)
plt.show()
