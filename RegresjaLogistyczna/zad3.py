import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel("dane.xlsx")
print(df)

df["Suma skumulowana"] = df["Liczba błędów"].cumsum()

print(df)

X = df["Nr miesiąca"].values.reshape(-1, 1)
y = df["Suma skumulowana"].values.reshape(-1, 1)

# Normalizacja danych (zakres [0, 1])
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_normalized = scaler_X.fit_transform(X)
y_normalized = scaler_y.fit_transform(y)

# Definicja funkcji logistycznej
def logistic_model(x, L, k, x0):
    """
    L - maksymalna wartość (asymptota),
    k - współczynnik wzrostu,
    x0 - punkt środkowy (miesiąc, w którym wzrost jest najszybszy).
    """
    return L / (1 + np.exp(-k * (x - x0)))

# Dopasowanie modelu logistycznego
popt, pcov = curve_fit(logistic_model, X_normalized.flatten(), y_normalized.flatten(), p0=[1, 1, 0.5])

# Parametry dopasowanego modelu
L, k, x0 = popt
print(f"Dopasowane parametry modelu logistycznego (znormalizowane):")
print(f"L (asymptota) = {L:.2f}")
print(f"k (współczynnik wzrostu) = {k:.4f}")
print(f"x0 (punkt środkowy) = {x0:.2f}")

# Predykcja na podstawie modelu
y_pred_normalized = logistic_model(X_normalized, *popt)

# Denormalizacja wyników
X_original = scaler_X.inverse_transform(X_normalized)
y_pred = scaler_y.inverse_transform(y_pred_normalized.reshape(-1, 1))

# Wizualizacja wyników
plt.figure(figsize=(10, 6))
plt.scatter(X_original, y, color='blue', label='Rzeczywiste dane')
plt.plot(X_original, y_pred, color='red', label='Model logistyczny')
plt.title('Model logistyczny (z normalizacją): Nr miesiąca vs Suma skumulowana błędów', fontsize=14)
plt.xlabel('Nr miesiąca', fontsize=12)
plt.ylabel('Suma błędów', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()


# plt.scatter(X, Y, marker="+")
# plt.show()