import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib.pyplot as plt

dane = pd.read_csv("dane.txt", delimiter="\t")
print(dane)

macierz_korelacji = dane.corr(method='pearson')
# print(macierz_korelacji)

korelacja_z_bf = macierz_korelacji["Pct.BF"].sort_values(ascending=False)
# print(korelacja_z_bf)

wybrane_predyktory = [item for item, value in korelacja_z_bf.items() if (np.abs(value) > 0.5 and item != "Pct.BF")]
# print(wybrane_predyktory)

tabela_wybranych_predyktorow = korelacja_z_bf.loc[wybrane_predyktory]
# print(tabela_wybranych_predyktorow)

vif_data = pd.DataFrame()
vif_data["Feature"] = wybrane_predyktory
vif_data["VIF"] = [variance_inflation_factor(dane[wybrane_predyktory].values, i) for i in range(len(wybrane_predyktory))]
print("VIF: ")
print(vif_data)

wybrane_predyktory.remove("Abdomen")
vif_data = pd.DataFrame()
vif_data["Feature"] = wybrane_predyktory
vif_data["VIF"] = [variance_inflation_factor(dane[wybrane_predyktory].values, i) for i in range(len(wybrane_predyktory))]
print("VIF: ")
print(vif_data)

wybrane_predyktory.remove("Hip")
vif_data = pd.DataFrame()
vif_data["Feature"] = wybrane_predyktory
vif_data["VIF"] = [variance_inflation_factor(dane[wybrane_predyktory].values, i) for i in range(len(wybrane_predyktory))]
print("VIF: ")
print(vif_data)

wybrane_predyktory.remove("Chest")
vif_data = pd.DataFrame()
vif_data["Feature"] = wybrane_predyktory
vif_data["VIF"] = [variance_inflation_factor(dane[wybrane_predyktory].values, i) for i in range(len(wybrane_predyktory))]
print("VIF: ")
print(vif_data)

wybrane_predyktory.remove("Thigh")
vif_data = pd.DataFrame()
vif_data["Feature"] = wybrane_predyktory
vif_data["VIF"] = [variance_inflation_factor(dane[wybrane_predyktory].values, i) for i in range(len(wybrane_predyktory))]
print("VIF: ")
print(vif_data)

wybrane_predyktory.remove("Waist")
vif_data = pd.DataFrame()
vif_data["Feature"] = wybrane_predyktory
vif_data["VIF"] = [variance_inflation_factor(dane[wybrane_predyktory].values, i) for i in range(len(wybrane_predyktory))]
print("VIF: ")
print(vif_data)

# Wybieramy dane tylko dla zmiennych z akceptowalnym VIF
final_features = ["Weight", "Density"]
X_final = dane[final_features]
y_final = dane["Pct.BF"]

# Regresja liniowa
reg = LinearRegression()
reg.fit(X_final, y_final)

# Wyniki
y_pred = reg.predict(X_final)
print("R^2 score:", r2_score(y_final, y_pred))
print("MSE:", mean_squared_error(y_final, y_pred))
print("Współczynniki regresji:", reg.coef_)
print("Intercept:", reg.intercept_)

plt.figure(figsize=(10, 6))
plt.scatter(y_final, y_pred, color='blue', label="Predykcja")
plt.plot([y_final.min(), y_final.max()], [y_final.min(), y_final.max()], color='red', linestyle='--', label="Idealna linia")
plt.xlabel("Rzeczywiste wartości Pct.BF")
plt.ylabel("Przewidywane wartości Pct.BF")
plt.title("Rzeczywiste vs Przewidywane wartości Body Fat (Pct.BF)")
plt.legend()
plt.savefig("WykresBF-VIF.png", dpi=300, transparent=None, bbox_inches='tight', pad_inches=0.1)
plt.show()
