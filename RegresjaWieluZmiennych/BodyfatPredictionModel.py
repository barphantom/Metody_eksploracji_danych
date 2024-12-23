import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("dane.txt", delimiter="\t")
# print(data.head())
# print(data.shape)

correlation_matrix = data.corr(method='pearson')
# print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix.abs(), annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.show()

target_variable = "Pct.BF"
correlation_with_target = data.corr()[target_variable].sort_values(ascending=False)
print("\nKorelacja ze zmienną zależną (Pct.BF):")
print(correlation_with_target)

# Wybór zmiennych silnie skorelowanych ze zmienną zależną (przykładowo |r| > 0.5)
selected_features = correlation_with_target[abs(correlation_with_target) > 0.5].index
selected_features = [feature for feature in selected_features if feature != target_variable]
print("\nWybrane zmienne do modelu (po korelacji):")
print(selected_features)
print(correlation_with_target[selected_features].values)
print(data[selected_features].values)


def calculate_vif(df, features):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features
    vif_data["VIF"] = [variance_inflation_factor(df[features].values, i) for i in range(len(features))]
    return vif_data


vif_data = calculate_vif(data, selected_features)
print("\nWspółczynniki VIF przed eliminacją kolinearności:")
print(vif_data)

while vif_data["VIF"].max() < 10:
    feature_to_remove = vif_data.sort_values("VIF", ascending=False).iloc[0]["Feature"]
    print(f"\nUsuwanie zmiennej: {feature_to_remove} z powodu wysokiego VIF ({vif_data['VIF'].max()})")
    selected_features.remove(feature_to_remove)
    vif_data = calculate_vif(data, selected_features)

selected_features.remove(vif_data.sort_values("VIF", ascending=False).iloc[0]["Feature"])
# selected_features.remove(vif_data.sort_values("VIF", ascending=False).iloc[1]["Feature"])
print("\nZmienne po eliminacji kolinearności:")
print(selected_features)

# 3. Przygotowanie danych do regresji
X = data[selected_features]
y = data[target_variable]

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Regresja liniowa
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Wyniki modelu
y_pred = model.predict(X_test)

# Wyświetlenie wyników
print("\nWspółczynniki regresji:")
for feature, coef in zip(selected_features, model.coef_):
    print(f"{feature}: {coef:.4f}")

print(f"\nWartość wyrazu wolnego (intercept): {model.intercept_:.4f}")
print(f"\nR² na zbiorze testowym: {r2_score(y_test, y_pred):.4f}")
print(f"Średni błąd kwadratowy (MSE): {mean_squared_error(y_test, y_pred):.4f}")
print(f"Pierwiastek z MSE (RMSE): {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

# 6. Wizualizacja predykcji vs rzeczywiste

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_pred), max(y_pred)], color='red', linestyle='--')
plt.xlabel("Rzeczywiste wartości (y_test)")
plt.ylabel("Przewidywane wartości (y_pred)")
plt.title("Regresja liniowa - rzeczywiste vs przewidywane")
plt.grid()
plt.show()
