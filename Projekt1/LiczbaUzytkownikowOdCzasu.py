import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns

# Rok, od i do którego mają być przewidywane dane
from_year = 2018
up_to_year = 2023

all_data = pd.read_csv("facebook_users.csv")

q4_data = all_data[all_data["Quarter"] == "Q4"]

q4_data = q4_data.copy()

X = q4_data[["Year"]]  # Zmienna niezależna - Rok
Y = q4_data["Users (in millions)"]  # Zmienna zależna - Liczba użytkowników

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

metrics = {
    "Współczynniki": ["Współczynnik regresji", "Współczynnik przesunięcia", "Błąd średniokwadratowy (MSE)",
                      "Współczynnik determinacji (R^2)"],
    "Wartości dla współczynników": [model.coef_[0], model.intercept_, mean_squared_error(Y_test, y_pred),
                                    r2_score(Y_test, y_pred)]
}

# Create a DataFrame from the dictionary
metrics_df = pd.DataFrame(metrics)

# Print the DataFrame
print(metrics_df)
print()

metrics_predictions = {
    "Dane": ["Rok", "Przewidywana liczba użytkowników", "Rzeczywista liczba użytkowników"],
    "Wartości": []
}

# Determine the last year in the actual data
last_actual_year = q4_data["Year"].max()

# Rozszerzenie lat do 2023
years_extended = pd.DataFrame({"Year": range(q4_data["Year"].min(), up_to_year + 1)})
predictions_extended = model.predict(years_extended)

# Przewidywane wartości dla lat po ostatnim roku rzeczywistym
predicted_years = years_extended[years_extended["Year"] >= from_year]
predicted_values = model.predict(predicted_years)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=q4_data["Year"], y=q4_data["Users (in millions)"], color="blue", label="Dane rzeczywiste", marker="o")

# Linia regresji dla rzeczywistych danych
plt.plot(years_extended[years_extended["Year"] <= last_actual_year]["Year"],
         predictions_extended[years_extended["Year"] <= last_actual_year], color="red", label="Linia regresji")

# Linia regresji dla przewidywanych danych (przerywana)
plt.plot(predicted_years["Year"], predicted_values, color="red", linestyle="--", label="Linia regresji (prognoza)",
         linewidth=1, alpha=0.5)

sns.scatterplot(x=predicted_years["Year"], y=predicted_values, color="green", label="Przewidywane punkty",
                marker="x", linewidth=2)

# Punkty rzeczywiste dla przewidywanych lat
real_data_for_predicted_years = pd.read_csv("Facebook_Users_Quarterly_Statista.csv")
real_data_for_predicted_years = real_data_for_predicted_years[real_data_for_predicted_years["Year"] > last_actual_year]
real_data_for_predicted_years = real_data_for_predicted_years[real_data_for_predicted_years["Quarter"] == "Q4"]
sns.scatterplot(x=real_data_for_predicted_years["Year"], y=real_data_for_predicted_years["Users (in millions)"],
                color="purple", label="Rzeczywiste dane dla przewidywanych lat", marker="o")

for lp, element in enumerate(predicted_values):
    metrics_predictions["Wartości"].append([predicted_years['Year'].iloc[lp], round(element),
                                            real_data_for_predicted_years["Users (in millions)"].iloc[lp]])

# convert to DataFrame
metrics_predictions_df = pd.DataFrame(metrics_predictions["Wartości"], columns=metrics_predictions["Dane"])
print(f"Analiza przewidywanych danych dla lat {last_actual_year + 1} - {up_to_year}")
print(f"Przewidywana liczba użytkowników podana jest w milionach")
print(metrics_predictions_df)

plt.title("Liczba użytkowników Facebooka w zależności od roku")
plt.xlabel("Rok")
plt.ylabel("Liczba użytkowników (w milionach)")
plt.legend()

plt.show()
