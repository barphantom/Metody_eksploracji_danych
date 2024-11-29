import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns

earnings = pd.read_csv("facebook_earnings.csv")

X = earnings[["Employment"]]
Y = earnings["Revenue (in millions)"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

print("Współczynniki regresji:", model.coef_)
print("Współczynnik przesunięcia (intercept):", model.intercept_)
print("Mean Squared Error (MSE):", mean_squared_error(Y_test, y_pred))
print("R^2 Score:", r2_score(Y_test, y_pred))


sns.scatterplot(x=earnings["Employment"], y=earnings["Revenue (in millions)"])
plt.plot(X, model.predict(earnings[["Employment"]]), color="red", label="Linia regresji")
plt.show()
