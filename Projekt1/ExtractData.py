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
q4_data.loc[:, "User Growth (in millions)"] = q4_data["Users (in millions)"].diff()
print(q4_data)
q4_data.loc[q4_data["Year"] == 2009, "User Growth (in millions)"] = 360
print(q4_data)

X = q4_data[["Year"]]
Y = q4_data["User Growth (in millions)"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

print("Współczynniki regresji:", model.coef_)
print("Współczynnik przesunięcia (intercept):", model.intercept_)
print("Mean Squared Error (MSE):", mean_squared_error(Y_test, y_pred))
print("R^2 Score:", r2_score(Y_test, y_pred))

future_years = pd.DataFrame({"Year": [2018, 2019, 2020, 2021, 2022]})
future_predictions = model.predict(future_years)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_train["Year"], y=Y_train, color="blue", label="Dane rzeczywiste", marker="+")
plt.plot(X_test["Year"], y_pred, color="red", label="Linia regresji")
plt.plot(future_years["Year"], future_predictions, color="green", linestyle="--", label="Prognoza (2018-2022)")


plt.title("Przyrost użytkowników Facebooka w zależności od roku")
plt.xlabel("Rok")
plt.ylabel("Przyrost użytkowników (w milionach)")
plt.legend()

plt.show()
