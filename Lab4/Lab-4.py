import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.lines as mlines

# Przygotowanie danych (przykład - dostosuj do swojego pliku)
dane = pd.read_excel("Płatki-sniadaniowe-cereals.xlsx")  # Załaduj dane z pliku Excel


def zaleznosc1(dane):
    # Wybieramy cechy (kalorie i tłuszcz) oraz etykiety
    X = dane[["kalorie", "tluszcz"]].values
    y = []

    # Tworzenie etykiet klas na podstawie kalorii i tłuszczu
    for calories, fat in X:
        if calories <= 110 and fat <= 2:
            y.append(0)  # Niskokaloryczne i niskotłuszczowe
        elif calories > 110 and fat <= 2:
            y.append(1)  # Wysokokaloryczne i niskotłuszczowe
        else:
            y.append(2)  # Wysokokaloryczne i wysokotłuszczowe

    # Konwersja etykiet na numpy array
    y = np.array(y)

    # Normalizacja cech
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Podział danych na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Trening modelu KNN
    k = 5  # Liczba sąsiadów
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X_train, y_train)

    # Predykcja na zbiorze testowym
    y_pred = knn.predict(X_test)

    # Ocena modelu
    print("Dokładność:", accuracy_score(y_test, y_pred))
    print("Raport klasyfikacji:\n", classification_report(y_test, y_pred, zero_division=0))
    print("Przewidywania modelu: ", y_pred)

    # Tworzenie siatki dla granic decyzyjnych
    x_min, x_max = X_scaled[:, 0].min() - 0.1, X_scaled[:, 0].max() + 0.1
    y_min, y_max = X_scaled[:, 1].min() - 0.1, X_scaled[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # Predykcja dla każdego punktu siatki
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Wykres granic decyzyjnych
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')

    # Definiujemy odcienie kolorów dla każdej klasy
    train_colors = ['#FF5733', '#3357FF', '#33FF57']  # Czerwony, zielony, niebieski
    test_colors = ['#C70039', '#1F618D', '#28B463']  # Ciemniejsze odcienie

    # Etykiety klas
    # labels = ['Niskokaloryczne i niskotłuszczowe', 'Wysokokaloryczne i niskotłuszczowe', 'Wysokokaloryczne i wysokotłuszczowe']
    labels = ['Klasa 1', 'Klasa 2', 'Klasa 3']

    # Rozkład danych treningowych
    for i, color in enumerate(train_colors):
        plt.scatter(
            X_train[y_train == i, 0],
            X_train[y_train == i, 1],
            color=color,
            marker='o',
            label=f"Dane treningowe: {labels[i]}",
            alpha=1.0
        )

    # Rozkład danych testowych z predykcjami
    for i, color in enumerate(test_colors):
        plt.scatter(
            X_test[y_pred == i, 0],
            X_test[y_pred == i, 1],
            color=color,
            marker='x',
            s=100,
            label=f"Predykcje testowe: {labels[i]}",
            alpha=1.0
        )

    # Definiujemy kolory tła dla legendy
    background_colors = ['#440154', '#21908d', '#fde725']  # Kolory z cmap='viridis'
    background_labels = ['Klasa 1 - Niskokaloryczne i niskotłuszczowe', 'Klasa 2- Wysokokaloryczne i niskotłuszczowe',
                         'Klasa 3 - Wysokokaloryczne i wysokotłuszczowe']

    # Tworzenie elementów do legendy
    background_patches = [
        mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=8, label=label)
        for color, label in zip(background_colors, background_labels)
    ]

    # Dodanie wszystkich elementów do legendy
    plt.legend(handles=background_patches + plt.legend().legend_handles, fontsize=8)

    plt.title(f"Granice decyzyjne KNN (k = {k})")
    plt.xlabel("Kalorie (znormalizowane)")
    plt.ylabel("Tłuszcz (znormalizowany)")
    plt.grid(True)
    plt.show()


def zaleznosc2(dane):
    # Model 2 - Zależność ilości cukru od występowania na półce nr 1
    x2 = np.array(dane["cukry"]).reshape(-1, 1)
    y2 = np.array(dane["polka_1"])

    # Trening modelu KNN
    n_neighbors = 3  # Liczba sąsiadów
    model2 = KNeighborsClassifier(n_neighbors)
    model2.fit(x2, y2)

    # Ocena modelu
    print(f"Dopasowanie modelu 2: {model2.score(x2, y2)}")

    # Wypisanie przewidywań dla wartości cukru w zakresie 0-20 z krokiem 1
    print(f"Przewidywania modelu 2:\n{model2.predict(np.arange(0, 20).reshape(-1, 1))}")

    # Wizualizacja wyników

    # Tworzenie siatki dla granic decyzyjnych
    x_min, x_max = x2.min() - 1, x2.max() + 1  # Ustawiamy zakres na podstawie rzeczywistych danych
    xx = np.arange(x_min, x_max, 0.01).reshape(-1, 1)

    # Predykcja dla każdego punktu siatki
    Z = model2.predict(xx)

    # Wykres granic decyzyjnych
    plt.figure(figsize=(10, 6))
    plt.scatter(x2, y2, c=y2, edgecolor='k', cmap=plt.cm.Paired, label="Dane")
    plt.plot(xx, Z, color="red", label="Granica decyzyjna")
    plt.title(f"Zależność ilości cukru od występowania na półce nr 1 (k = {n_neighbors})")
    plt.xlabel("Cukry")
    plt.ylabel("Półka nr 1 (0 = nie, 1 = tak)")
    plt.legend()
    plt.grid(True)
    plt.show()


def zaleznosc3(dane):
    # Tworzenie klasy binarnej dla potasu (1 = potas > 180, 0 = potas <= 180)
    dane["potas"] = dane["potas"] > 180
    dane["potas"] = dane["potas"].astype("category").cat.codes

    # Przygotowanie danych wejściowych i wyjściowych
    x3 = np.array(dane["cukry"]).reshape(-1, 1)  # Ilość cukrów jako cecha wejściowa
    y3 = np.array(dane["potas"])  # Klasa binarna dla potasu

    # Trenowanie modelu KNN bez normalizacji
    k = 3
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x3, y3)

    # Siatka punktów do wizualizacji granicy decyzyjnej
    x_grid = np.linspace(min(x3), max(x3), 100).reshape(-1, 1)  # Zmieniamy zakres na rzeczywisty zakres cukrów
    y_grid_pred = knn.predict(x_grid)

    print(f"Dopasowanie modelu: {knn.score(x3, y3)}")
    print(f"Przewidywania modelu:\n{knn.predict(np.arange(0, 20).reshape(-1, 1))}")

    # Wizualizacja
    plt.figure(figsize=(8, 6))
    plt.scatter(x3, y3, color='blue', edgecolor='k', label='Dane')
    plt.plot(x_grid, y_grid_pred, color='red', label='Granica decyzyjna')
    plt.title(f"Zależność występowania cukru od ilości potasu (k = {k})")
    plt.xlabel("Cukry")
    plt.ylabel("Potas > 180 (0 = nie, 1 = tak)")
    plt.legend()
    plt.grid()
    plt.show()

print("\n============================================================")
print("Zależność 1:")
zaleznosc1(dane)
print("\n============================================================")
print("Zależność 2:")
zaleznosc2(dane)
print("\n============================================================")
print("Zależność 3:")
zaleznosc3(dane)
