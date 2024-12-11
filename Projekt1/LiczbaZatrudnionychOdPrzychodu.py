import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# Funkcja do wyboru lat
def wybierz_okres():
    print("Podaj rok, od którego mają być przewidywane dane:")
    print("1 - 2018 - 2020")
    print("2 - 2021 - 2023")
    print("3 - 2018 - 2023")
    print("4 - 2024 - 2027")
    choice = input("Wybierz opcję: ")

    if choice == "1":
        return 2018, 2020, "2018 - 2020", "2009 - 2017"
    elif choice == "2":
        return 2021, 2023, "2021 - 2023", "2009 - 2020"
    elif choice == "3":
        return 2018, 2023, "2018 - 2023", "2009 - 2017"
    elif choice == "4":
        return 2024, 2027, "2024 - 2027", "2009 - 2023"
    else:
        print("Niepoprawny wybór. Wybrano domyślnie opcję 2.")
        return 2021, 2023, "2021 - 2023", "2009 - 2020"

# Funkcja do wczytania i przygotowania danych
def przygotuj_dane(filepath, analyse_from_year, from_year):
    data = pd.read_csv(filepath)
    data_filtered = data[(data["rok"] >= analyse_from_year) & (data["rok"] < from_year)]
    return data_filtered[["przychod_mln"]], data_filtered["liczba_zatrudnionych"], data

# Funkcja do trenowania modelu liniowego
def trenowanie_modelu(X_train, Y_train):
    model = LinearRegression()
    model.fit(X_train, Y_train)
    return model

# Funkcja do oceny modelu
def ocen_model(model, X_test, Y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(Y_test, y_pred)
    r2 = r2_score(Y_test, y_pred)
    return model.coef_[0], model.intercept_, mse, r2

# Funkcja do rozszerzenia lat dla prognozy
def prognozuj(model, X, from_year, up_to_year, real_data):
    years_extended = pd.DataFrame({"rok": range(from_year, up_to_year + 1)})
    predicted_years_data = pd.DataFrame({"przychod_mln": [X["przychod_mln"].mean()] * len(years_extended)})

    predicted_values = model.predict(predicted_years_data)
    return years_extended["rok"], predicted_values, real_data[
        (real_data["rok"] >= from_year) & (real_data["rok"] <= up_to_year)
        ]

# Funkcja do wizualizacji
def wizualizacja(model, X, Y, predicted_years, predicted_values, real_data, from_year, up_to_year, data_years, predict_years):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X["przychod_mln"], y=Y, color="blue", label="Liczba zatrudnionych", marker="o")

    # Linia regresji rzeczywistej
    predictions_extended = model.predict(X)
    plt.plot(X["przychod_mln"], predictions_extended, color="red", label="Linia regresji rzeczywista")

    # Rzeczywiste dane dla prognozowanego przedziału
    if not real_data.empty:
        sns.scatterplot(x=real_data["przychod_mln"], y=real_data["liczba_zatrudnionych"], color="purple",
                        label="Rzeczywiste dane (dla prognozy)", marker="o")

        X_regression_line_predicted = [X["przychod_mln"].max(), real_data["przychod_mln"].max()]
        # prognoza Y z X
        Y_regression_line_predicted = [model.coef_[0] * X_regression_line_predicted[0] + model.intercept_,
                                       model.coef_[0] * X_regression_line_predicted[1] + model.intercept_]
        plt.plot(X_regression_line_predicted, Y_regression_line_predicted, color="green", linestyle="--", label="Linia regresji prognozowana")

    if predict_years != "2024 - 2027":
        plt.title(
            f"Liczba zatrudnionych w latach {data_years} w zależności od przychodu i prognoza na lata {predict_years}\nze sprawdzeniem zgodności prognoz z rzeczywistymi danymi")
    else:
        plt.title(
            f"Liczba zatrudnionych w latach {data_years} w zależności od przychodu")

    plt.xlabel("Przychód (mln)")
    plt.ylabel("Liczba zatrudnionych")
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
    plt.show()

# Główna funkcja programu
def main():
    filepath = "dane_med_lab1.csv"
    analyse_from_year = 2009
    from_year, up_to_year, predict_string_years, data_string_years = wybierz_okres()

    X, Y, real_data = przygotuj_dane(filepath, analyse_from_year, from_year)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = trenowanie_modelu(X_train, Y_train)
    coef, intercept, mse, r2 = ocen_model(model, X_test, Y_test)

    metrics = {
        "Współczynniki": ["Współczynnik regresji", "Współczynnik przesunięcia", "Błąd średniokwadratowy (MSE)",
                          "Współczynnik determinacji (R^2)"],
        "Wartości dla współczynników": [coef, intercept, mse, r2]
    }

    metrics_df = pd.DataFrame(metrics)
    print(metrics_df)

    predicted_years, predicted_values, real_predicted_data = prognozuj(model, X, from_year, up_to_year, real_data)
    wizualizacja(model, X, Y, predicted_years, predicted_values, real_predicted_data, from_year, up_to_year,
                 data_string_years, predict_string_years)

if __name__ == "__main__":
    main()