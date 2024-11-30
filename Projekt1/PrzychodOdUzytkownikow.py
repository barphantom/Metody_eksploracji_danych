import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

earnings = pd.read_csv("facebook_earnings.csv")
users = pd.read_csv("facebook_users.csv")

earnings.drop([0, 1], axis=0, inplace=True)

q4_data = users[users["Quarter"] == "Q4"]

merged = pd.merge(earnings, q4_data, on="Year")

X = merged[["Users (in millions)"]]
Y = merged["Revenue (in millions)"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, Y_train)

log_Y = np.log(Y)
exponential_model = LinearRegression()
exponential_model.fit(X, log_Y)

a1 = exponential_model.coef_[0]  # log(a)
a0 = exponential_model.intercept_  # log(b)
a = np.exp(a1)
b = np.exp(a0)

# Generate a range of values for a smooth curve
X_range = pd.DataFrame(np.linspace(X.min(), X.max(), 100), columns=["Users (in millions)"])
y_pred_exp_smooth = b * a ** X_range.values.flatten()

# Predictions for the entire range
y_pred_range = model.predict(X_range)



plt.figure(figsize=(10, 6))
sns.scatterplot(x=merged["Users (in millions)"], y=merged["Revenue (in millions)"], label="Rzeczywiste dane", marker="o")
plt.plot(X_range, y_pred_exp_smooth, color="green", label="Exponential regresji")
plt.plot(X_range, y_pred_range, color="red", label="Regresja liniowa")


plt.title("Regresja: Przychód od liczby użytkowników Facebooka")
plt.xlabel("Liczba użytkowników (w milionach)")
plt.ylabel("Przychód (w milionach $)")
plt.legend()

plt.show()