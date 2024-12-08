import pandas as pd
import numpy as np
import statsmodels.api as sm
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, d2_pinball_score
import matplotlib.pyplot as plt

# Wczytanie danych
dane = pd.read_csv("dane.txt", delimiter="\t")
print(dane)

# Macierz korelacji i wstępna selekcja predyktorów
macierz_korelacji = dane.corr(method='pearson')
korelacja_z_bodyfat = macierz_korelacji["Pct.BF"].sort_values(ascending=False)

# Wybór początkowych predyktorów na podstawie korelacji
wybrane_predyktory = [item for item, value in korelacja_z_bodyfat.items() if (np.abs(value) > 0.5 and item != "Pct.BF")]

# Dodanie zmiennej zależnej i stałej dla modelu
X = dane[wybrane_predyktory]
y = dane["Pct.BF"]
X = sm.add_constant(X)

# Backward elimination
def backward_elimination(X, y, significance_level=0.05):
    """
    Funkcja realizująca backward elimination.
    """
    model = sm.OLS(y, X).fit()
    while True:
        max_p_value = max(model.pvalues)
        if max_p_value > significance_level:
            excluded_feature = model.pvalues.idxmax()  # Nazwa zmiennej do usunięcia
            print(f"Usuwam zmienną: {excluded_feature} (p-value = {max_p_value})")
            X = X.drop(columns=[excluded_feature])
            model = sm.OLS(y, X).fit()
        else:
            break
    return model, X

# Przeprowadzenie backward elimination
model, X_selected = backward_elimination(X, y)

# Wyniki backward elimination
print("\nOstateczne predyktory:")
print(X_selected.columns)
print("\nPodsumowanie modelu:")
print(model.summary())

# Regresja liniowa z wybranymi predyktorami
X_final = X_selected.drop(columns=["const"])  # Usunięcie kolumny stałej dla scikit-learn
reg = LinearRegression()
reg.fit(X_final, y)

# Wyniki regresji
y_pred = reg.predict(X_final)
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
plt.savefig("WykresBF-Backward.png", dpi=300, transparent=None, bbox_inches='tight', pad_inches=0.1)
plt.show()
