import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


# Funkcja do wyboru okresu prognozy
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


# Wczytaj i przygotuj dane
def przygotuj_dane(filepath, analyse_from_year, from_year):
    data = pd.read_csv(filepath)
    data_filtered = data[(data["rok"] >= analyse_from_year) & (data["rok"] < from_year)]
    return data_filtered[["liczba_uzytkonikow_mln"]], data_filtered["przychod_mln"], data


def metryka_modelu(model, X_test, Y_test, y_pred):
    # Obliczenie metryk oceny modelu
    metrics = {
        "Współczynniki": ["Współczynnik regresji", "Współczynnik przesunięcia", "Błąd średniokwadratowy (MSE)",
                          "Współczynnik determinacji (R^2)"],
        "Wartości dla współczynników": [
            model.coef_[0],  # Współczynnik regresji
            model.intercept_,  # Współczynnik przesunięcia
            mean_squared_error(Y_test, y_pred),  # Błąd średniokwadratowy (MSE)
            r2_score(Y_test, y_pred)  # Współczynnik determinacji (R^2)
        ]
    }

    # Utworzenie DataFrame z metrykami
    metrics_df = pd.DataFrame(metrics)

    # Wyświetlenie metryk
    print(metrics_df)


# Główna funkcja
def main():
    filepath = "dane_med_lab1.csv"
    analyse_from_year = 2009
    from_year, up_to_year, predict_string_years, data_string_years = wybierz_okres()

    X, Y, real_data = przygotuj_dane(filepath, analyse_from_year, from_year)
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

    # Generowanie zakresu wartości
    X_range = pd.DataFrame(np.linspace(X.min(), X.max(), 100), columns=["liczba_uzytkonikow_mln"])
    y_pred_exp_smooth = b * a ** X_range.values.flatten()

    # Regresja wielomianowa
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    poly_model = LinearRegression()
    poly_model.fit(X_poly, Y)

    # Generowanie zakresu wartości
    X_range_poly = poly.fit_transform(X_range)

    # Prognozy dla całego zakresu
    y_pred_poly_smooth = poly_model.predict(X_range_poly)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X["liczba_uzytkonikow_mln"], y=Y, label="Rzeczywiste dane", marker="o")
    plt.plot(X_range, y_pred_exp_smooth, color="green", label="Model wykładniczy", alpha=0.25)
    plt.plot(X_range, y_pred_poly_smooth, color="blue", label="Model wielomianowy", alpha=0.25)

    # Definiowanie wag dla funkcji wykładniczej i wielomianowej
    weight_exp = 0.5
    weight_poly = 0.5

    # Sprawdzenie, czy wagi sumują się do 1
    assert weight_exp + weight_poly == 1, "Wagi muszą sumować się do 1"

    # Obliczanie średniej ważonej prognoz
    y_pred_weighted = weight_exp * y_pred_exp_smooth + weight_poly * y_pred_poly_smooth
    plt.plot(X_range, y_pred_weighted, color="purple", label="Superpozycja dwóch modeli", alpha=0.7)

    # Dopasowanie modelu regresji liniowej do średniej ważonej
    linear_model_weighted = LinearRegression()
    linear_model_weighted.fit(X_range, y_pred_weighted)

    # Prognozy dla całego zakresu przy użyciu modelu liniowego na średniej ważonej
    y_pred_linear_weighted = linear_model_weighted.predict(X_range)

    plt.plot(X_range, y_pred_linear_weighted, color="orange", label="Regresja liniowa (średnia ważona)")

    plt.title(
        f"Regresja: Przychód od liczby użytkowników Facebooka ({data_string_years})")
    plt.xlabel("Liczba użytkowników (w milionach)")
    plt.ylabel("Przychód (w milionach $)")
    plt.legend()

    plt.show()

    # Przewidywanie wartości na zbiorze testowym dla regresji liniowej z modelu średniej ważonej
    y_pred_linear_weighted_test = linear_model_weighted.predict(X_test)

    # Wywołanie funkcji metryka_modelu dla regresji liniowej z modelu średniej ważonej
    print()
    metryka_modelu(linear_model_weighted, X_test, Y_test, y_pred_linear_weighted_test)


if __name__ == "__main__":
    main()
