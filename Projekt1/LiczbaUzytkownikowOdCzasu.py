import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns

all_data = pd.read_csv("facebook_users.csv")
print(all_data)

q4_data = all_data[all_data["Quarter"] == "Q4"]
print(q4_data)

q4_data = q4_data.copy()

X = q4_data[["Year"]]  # Zmienna niezależna - Rok
Y = q4_data["Users (in millions)"]  # Zmienna zależna - Liczba użytkowników
print(X.shape)
print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X_train)
print(X_test)

model = LinearRegression()
model.fit(X_train, Y_train)


y_pred = model.predict(X_test)

print("Współczynniki regresji:", model.coef_)
print("Współczynnik przesunięcia (intercept):", model.intercept_)
print("Mean Squared Error (MSE):", mean_squared_error(Y_test, y_pred))
print("R^2 Score:", r2_score(Y_test, y_pred))

years_extended = pd.DataFrame({"Year": range(q4_data["Year"].min(), q4_data["Year"].max() + 4)})
predictions_extended = model.predict(years_extended)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=q4_data["Year"], y=q4_data["Users (in millions)"], color="blue", label="Dane rzeczywiste", marker="o")
plt.plot(years_extended["Year"], predictions_extended, color="red", label="Linia regresji")

plt.title("Liczba użytkowników Facebooka w zależności od roku")
plt.xlabel("Rok")
plt.ylabel("Liczba użytkowników (w milionach)")
plt.legend()

plt.show()
