import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns

earnings = pd.read_csv("facebook_earnings.csv")
users = pd.read_csv("facebook_users.csv")

print(earnings)

earnings.drop([0, 1], axis=0, inplace=True)
print(earnings)

q4_data = users[users["Quarter"] == "Q4"]
print(q4_data)

merged = pd.merge(earnings, q4_data, on="Year")
print(merged)

X = merged[["Users (in millions)"]]
Y = merged["Revenue (in millions)"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

log_Y = np.log(Y)
exponential_model = LinearRegression()
exponential_model.fit(X, log_Y)

a1 = exponential_model.coef_[0]                     # log(a)
a0 = exponential_model.intercept_                   # log(b)
a = np.exp(a1)
b = np.exp(a0)

print(X.values)
print(X.values.flatten())
y_pred_exp = b * a ** X.values.flatten()
# y_pred_exp = a1 * X.values.flatten() + a0

# print("Współczynniki regresji:", model.coef_)
# print("Współczynnik przesunięcia (intercept):", model.intercept_)
# print("Mean Squared Error (MSE):", mean_squared_error(Y_test, y_pred))
# print("R^2 Score:", r2_score(Y_test, y_pred))


plt.figure(figsize=(10, 6))
sns.scatterplot(x=merged["Users (in millions)"], y=merged["Revenue (in millions)"], label="Rzeczywiste dane", marker="o")
plt.plot(merged["Users (in millions)"], model.predict(merged[["Users (in millions)"]]), color="red", label="Linia regresji")
plt.plot(merged["Users (in millions)"], y_pred_exp, color="green", label="Exponential regresji")


plt.title("Regresja: Przychód od liczby użytkowników Facebooka")
plt.xlabel("Liczba użytkowników (w milionach)")
plt.ylabel("Przychód (w milionach $)")
plt.legend()

plt.show()
