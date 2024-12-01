import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Wyświetlenie opcji wyboru lat do przewidywania danych
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

# Ustawienie zakresu lat na podstawie wyboru użytkownika
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

# Wczytanie danych z pliku CSV
all_data_csv = pd.read_csv("dane_med_lab1.csv")
all_data = all_data_csv.copy()

# Filtracja danych do analizy (dane roczne)
all_data = all_data[(all_data["rok"] >= analyse_from_year) & (all_data["rok"] < from_year)]

# Przygotowanie zmiennych niezależnych (X) i zależnych (Y)
X = all_data[["rok"]]  # Zmienna niezależna - Rok
Y = all_data["liczba_uzytkonikow_mln"]  # Zmienna zależna - Liczba użytkowników

# Podział danych na zbiór treningowy i testowy
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Trenowanie modelu regresji liniowej
model = LinearRegression()
model.fit(X_train, Y_train)

# Przewidywanie wartości na zbiorze testowym
y_pred = model.predict(X_test)

# Obliczenie metryk oceny modelu
metrics = {
    "Współczynniki": ["Współczynnik regresji", "Współczynnik przesunięcia", "Błąd średniokwadratowy (MSE)",
                      "Współczynnik determinacji (R^2)"],
    "Wartości dla współczynników": [model.coef_[0], model.intercept_, mean_squared_error(Y_test, y_pred),
                                    r2_score(Y_test, y_pred)]
}

# Utworzenie DataFrame z metrykami
metrics_df = pd.DataFrame(metrics)

# Wyświetlenie metryk
print(metrics_df)
print()

# Przygotowanie danych do prognozy
metrics_predictions = {
    "Dane": ["Rok", "Przewidywana liczba użytkowników", "Rzeczywista liczba użytkowników"],
    "Wartości": []
}

# Określenie ostatniego roku w rzeczywistych danych
last_actual_year = from_year - 1

# Rozszerzenie lat do prognozy
years_extended = pd.DataFrame({"rok": range(all_data["rok"].min(), up_to_year + 1)})
predictions_extended = model.predict(years_extended)

# Przewidywane wartości dla lat po ostatnim roku rzeczywistym
predicted_years = years_extended[years_extended["rok"] >= from_year]
predicted_values = model.predict(predicted_years)

# Wizualizacja danych
plt.figure(figsize=(10, 6))
sns.scatterplot(x=all_data["rok"], y=all_data["liczba_uzytkonikow_mln"], color="blue", label="Liczba użytkowników",
                marker="o")

# Linia regresji dla rzeczywistych danych
plt.plot(years_extended[years_extended["rok"] <= last_actual_year]["rok"],
         predictions_extended[years_extended["rok"] <= last_actual_year], color="red", label="Linia regresji")

# Linia regresji dla prognozowanych danych
plt.plot(predicted_years["rok"], predicted_values, color="red", linestyle="--", label="Linia regresji (prognoza)",
         linewidth=1, alpha=0.5)

# Wizualizacja prognozowanych wartości
sns.scatterplot(x=predicted_years["rok"], y=predicted_values, color="green", label="Prognozowana liczba użytkowników",
                marker="x", linewidth=2)

# Wizualizacja rzeczywistych danych dla prognozowanych lat
real_data_for_predicted_years = all_data_csv.copy()
real_data_for_predicted_years = real_data_for_predicted_years[
    (real_data_for_predicted_years["rok"] >= from_year) & (real_data_for_predicted_years["rok"] <= up_to_year)]
sns.scatterplot(x=real_data_for_predicted_years["rok"], y=real_data_for_predicted_years["liczba_uzytkonikow_mln"],
                color="purple", label="Rzeczywista liczba użytkowników (dla prognozy)", marker="o")

# Dodanie przewidywanych wartości do metryk
for lp, element in enumerate(predicted_values):
    if lp < len(real_data_for_predicted_years):
        metrics_predictions["Wartości"].append([predicted_years['rok'].iloc[lp], round(element),
                                                real_data_for_predicted_years["liczba_uzytkonikow_mln"].iloc[lp]])
    else:
        metrics_predictions["Wartości"].append([predicted_years['rok'].iloc[lp], round(element), None])

# Utworzenie DataFrame z przewidywanymi wartościami
metrics_predictions_df = pd.DataFrame(metrics_predictions["Wartości"], columns=metrics_predictions["Dane"])
print(f"Analiza przewidywanych danych dla lat {last_actual_year + 1} - {up_to_year}")
print(f"Przewidywana liczba użytkowników podana jest w milionach")
print(metrics_predictions_df)

# Ustawienia wykresu
plt.title(
    f"Liczba użytkowników w latach {data_string_years} i predykcja na lata {predict_string_years} \n"
    f"ze sprawdzeniem zgodności prognoz z rzeczywistymi danymi")
plt.xlabel("Rok")
plt.ylabel("Liczba użytkowników (w milionach)")
plt.legend()

# Dodanie formatowania osi X, aby wyświetlać maksymalnie 9 podziałek
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))

# Wyświetlenie wykresu
plt.show()