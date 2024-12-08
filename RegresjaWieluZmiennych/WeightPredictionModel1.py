import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt

# Wczytanie danych
dane = pd.read_csv("dane.txt", delimiter="\t")
print(dane)

# Obliczenie macierzy korelacji
macierz_korelacji = dane.corr(method='pearson')

# Wyznaczenie zmiennych z wysoką korelacją z "Weight"
korelacja_z_waga = macierz_korelacji["Weight"].sort_values(ascending=False)
print("Korelacja z 'Weight':")
print(korelacja_z_waga)

# Wybór predyktorów na podstawie korelacji (wartości bezwzględne > 0.5 i różne od 'Weight')
wybrane_predyktory = [item for item, value in korelacja_z_waga.items() if (np.abs(value) > 0.5 and item != "Weight")]
print("Wybrane predyktory:", wybrane_predyktory)

# Obliczenie VIF dla wybranych predyktorów
def oblicz_vif(data, features):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features
    X = data[features].values
    vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    return vif_data

# Początkowy VIF
vif_data = oblicz_vif(dane, wybrane_predyktory)
print("Początkowe VIF:")
print(vif_data)

# Iteracyjne usuwanie predyktorów z wysokim VIF
while vif_data["VIF"].max() > 500:
    najwiekszy_vif = vif_data.loc[vif_data["VIF"].idxmax(), "Feature"]
    wybrane_predyktory.remove(najwiekszy_vif)
    vif_data = oblicz_vif(dane, wybrane_predyktory)
    print("Zaktualizowane VIF po usunięciu:", najwiekszy_vif)
    print(vif_data)

# Finalny zestaw predyktorów
print("Finalne predyktory:", wybrane_predyktory)

# Przygotowanie danych do regresji
X_final = dane[wybrane_predyktory]
y_final = dane["Weight"]

# Regresja liniowa
reg = LinearRegression()
reg.fit(X_final, y_final)

# Wyniki modelu
y_pred = reg.predict(X_final)
print("R^2 score:", r2_score(y_final, y_pred))
print("MSE:", mean_squared_error(y_final, y_pred))
print("Współczynniki regresji:", reg.coef_)
print("Intercept:", reg.intercept_)

# Wykres rzeczywiste vs przewidywane
plt.figure(figsize=(10, 6))
plt.scatter(y_final, y_pred, color='blue', label="Predykcja")
plt.plot([y_final.min(), y_final.max()], [y_final.min(), y_final.max()], color='red', linestyle='--', label="Idealna linia")
plt.xlabel("Rzeczywiste wartości Weight")
plt.ylabel("Przewidywane wartości Weight")
plt.title("Rzeczywiste vs Przewidywane wartości Weight")
plt.legend()
plt.savefig("Wykres-Weight-VIF.png", dpi=300, transparent=None, bbox_inches='tight', pad_inches=0.1)
plt.show()
