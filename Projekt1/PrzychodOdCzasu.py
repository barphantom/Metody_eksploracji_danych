import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns

# Rok, od i do którego mają być przewidywane dane
print("Podaj rok, od którego mają być przewidywane dane:")
print("1 - 2018 - 2020")
print("2 - 2021 - 2023")
print("3 - 2018 - 2023")
print("4 - 2024 - 2027")
choice = input("Wybierz opcję: ")
predict_string_years = ""
data_string_years = ""
# Rok, od którego zacząć analizę danych
analyse_from_year = 2009

if choice == "1":
    from_year = 2018
    up_to_year = 2020
    predict_string_years = "2018 - 2020"
    data_string_years = "2009 - 2017"
elif choice == "2":
    from_year = 2021
    up_to_year = 2023
    predict_string_years = "2021 - 2023"
    data_string_years = "2009 - 2020"
elif choice == "3":
    from_year = 2018
    up_to_year = 2023
    predict_string_years = "2018 - 2023"
    data_string_years = "2009 - 2017"
elif choice == "4":
    from_year = 2024
    up_to_year = 2027
    predict_string_years = "2024 - 2027"
    data_string_years = "2009 - 2023"
else:
    print("Niepoprawny wybór. Wybrano opcję 2.")
    from_year = 2021
    up_to_year = 2023

# Wczytanie danych
earnings_data = pd.read_csv("dane_med_lab1.csv")
earnings = earnings_data.copy()

earnings = earnings[(earnings["rok"] >= analyse_from_year) & (earnings["rok"] < from_year)]

# Przygotowanie danych
X = earnings[["rok"]]  # Zmienna niezależna - Rok
Y = earnings["przychod_mln"]  # Zmienna zależna - Przychody

# Podział danych na zbiór treningowy i testowy
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Tworzenie modelu regresji liniowej
linear_model = LinearRegression()
linear_model.fit(X_train, Y_train)

# Przewidywanie na zbiorze testowym
y_pred_linear = linear_model.predict(X_test)

# Obliczanie metryk dla modelu liniowego
metrics_linear = {
    "Współczynniki": ["Współczynnik regresji", "Współczynnik przesunięcia", "Błąd średniokwadratowy (MSE)", "Współczynnik determinacji (R^2)"],
    "Wartości dla współczynników": [linear_model.coef_[0], linear_model.intercept_, mean_squared_error(Y_test, y_pred_linear), r2_score(Y_test, y_pred_linear)]
}

# Determine the last year in the actual data
last_actual_year = from_year - 1

# Rozszerzenie lat do 2023
years_extended = pd.DataFrame({"rok": range(earnings["rok"].min(), up_to_year + 1)})
predictions_extended = linear_model.predict(years_extended)

# Przewidywane wartości dla lat po ostatnim roku rzeczywistym
predicted_years = years_extended[years_extended["rok"] >= from_year]
predicted_values = linear_model.predict(predicted_years)

# Tworzenie DataFrame z metrykami dla modelu liniowego
metrics_linear_df = pd.DataFrame(metrics_linear)
print("Metryki dla modelu liniowego:")
print(metrics_linear_df)
print()

# Tworzenie modelu regresji wykładniczej
log_Y = np.log(Y)
exponential_model = LinearRegression()
exponential_model.fit(X, log_Y)

# Przewidywanie na zbiorze testowym dla modelu wykładniczego
y_pred_exp_log = exponential_model.predict(X_test)
y_pred_exp = np.exp(y_pred_exp_log)

# Obliczanie metryk dla modelu wykładniczego
metrics_exp = {
    "Współczynniki": ["Współczynnik regresji", "Współczynnik przesunięcia", "Błąd średniokwadratowy (MSE)", "Współczynnik determinacji (R^2)"],
    "Wartości dla współczynników": [exponential_model.coef_[0], exponential_model.intercept_, mean_squared_error(Y_test, y_pred_exp), r2_score(Y_test, y_pred_exp)]
}

# Tworzenie DataFrame z metrykami dla modelu wykładniczego
metrics_exp_df = pd.DataFrame(metrics_exp)
print("Metryki dla modelu wykładniczego:")
print(metrics_exp_df)
print()

# Generowanie zakresu wartości dla gładkiej krzywej
X_range = pd.DataFrame(np.linspace(X.min(), X.max(), 100), columns=["rok"])
y_pred_exp_smooth = np.exp(exponential_model.predict(X_range))

# Wizualizacja danych
plt.figure(figsize=(10, 6))
sns.scatterplot(x=earnings["rok"], y=earnings["przychod_mln"], color="blue", label="Przychód", marker="o")

# Linia regresji dla rzeczywistych danych (model liniowy)
plt.plot(years_extended[years_extended["rok"] <= earnings["rok"].max()]["rok"], predictions_extended[years_extended["rok"] <= earnings["rok"].max()], color="red", label="Linia regresji (liniowa)")

# Linia regresji dla przewidywanych danych (przerywana)
plt.plot(predicted_years["rok"], predicted_values, color="red", linestyle="--", label="Linia regresji (prognoza)",
         linewidth=1, alpha=0.5)
# Linia ekspotencjalna
plt.plot(X_range["rok"], y_pred_exp_smooth, color="purple", linestyle="--", label="Linia regresji (ekspotencjalna)", alpha=0.5)
sns.scatterplot(x=predicted_years["rok"], y=predicted_values, color="green", label="Prognozowana liczba użytkowników",
                marker="x", linewidth=2)


plt.title(f"Przychody w latach {data_string_years} i predykcja na lata {predict_string_years} \n ze sprawdzeniem zgodności prognoz z rzeczywistymi danymi")
plt.xlabel("Rok")
plt.ylabel("Przychody (w milionach $)")
plt.legend()

plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))

# Wypisz przewidziany przychód dla lat po ostatnim roku rzeczywistym
predicted_values_df = pd.DataFrame({"rok": predicted_years["rok"], "przychod_mln": predicted_values})
print(f"Przewidywane przychody dla lat {last_actual_year + 1} - {up_to_year}")
print(predicted_values_df)

plt.show()