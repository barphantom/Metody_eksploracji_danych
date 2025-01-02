import pandas as pd

dane_surowe = pd.read_csv("Lab_3-Zad_2.csv", delimiter=";")

# Usunięcie ostatniego wiersza z danymi
dane = dane_surowe.drop(dane_surowe.index[-1])

# Zliczenie tak i nie dla kolumny klasyfikacja_spam
tak = dane[dane["klasyfikacja_spam"] == "tak"].shape[0]
nie = dane[dane["klasyfikacja_spam"] == "nie"].shape[0]
print(f"Rozkład klasyfikacji:")
print(f"Tak: {tak}, Nie: {nie}")

# Wyliczenie prawdopoobieństwa tak oraz nie dla kolumny klasyfikacja_spam
prawd_tak = tak / dane.shape[0]
prawd_nie = nie / dane.shape[0]
print(f"P(tak): {prawd_tak}, P(nie): {prawd_nie}")

def prawdopodobienstwo_warunkowe(dane, kolumna):
    "Wyliczenie prawdopodobieństwa warunkowego dla podanej kolumny"
    prawdopodobienstwo_tak_tak = dane[(dane["klasyfikacja_spam"] == "tak") & (dane[kolumna] == "tak")].shape[0] / tak
    prawdopodobienstwo_nie_tak = dane[(dane["klasyfikacja_spam"] == "nie") & (dane[kolumna] == "tak")].shape[0] / nie
    prawdopodobienstwo_tak_nie = dane[(dane["klasyfikacja_spam"] == "tak") & (dane[kolumna] == "nie")].shape[0] / tak
    prawdopodobienstwo_nie_nie = dane[(dane["klasyfikacja_spam"] == "nie") & (dane[kolumna] == "nie")].shape[0] / nie
    print(f"P(tak|tak): {prawdopodobienstwo_tak_tak}")
    print(f"P(tak|nie): {prawdopodobienstwo_nie_tak}")
    print(f"P(nie|tak): {prawdopodobienstwo_tak_nie}")
    print(f"P(nie|nie): {prawdopodobienstwo_nie_nie}")
    return prawdopodobienstwo_tak_tak, prawdopodobienstwo_nie_tak, prawdopodobienstwo_tak_nie, prawdopodobienstwo_nie_nie

# Wyznacz prawdopodobieństwa warunkowe dla kolumn pieniadz, darmowy, bogaty, nieprzyzwoicie, tajny
prawdopodobienstwa = {}
print("\nPrawdopodobieństwa warunkowe:")
for kolumna in dane.columns[1:-1]:
    print(f"\n{kolumna}")
    prawdopodobienstwa[kolumna] = prawdopodobienstwo_warunkowe(dane, kolumna)


# Pobranie ostatniego wiersza
ostatni_wiersz = dane_surowe.iloc[-1]

# Wyliczenie prawdopodobieństwa dla klasy "tak"
p_tak = prawd_tak
for kolumna in dane.columns[1:-1]:
    if ostatni_wiersz[kolumna] == "tak":
        p_tak *= prawdopodobienstwa[kolumna][0]  # P(atrybut=tak|spam=tak)
    else:
        p_tak *= prawdopodobienstwa[kolumna][2]  # P(atrybut=nie|spam=tak)

# Wyliczenie prawdopodobieństwa dla klasy "nie"
p_nie = prawd_nie
for kolumna in dane.columns[1:-1]:
    if ostatni_wiersz[kolumna] == "tak":
        p_nie *= prawdopodobienstwa[kolumna][1]  # P(atrybut=tak|spam=nie)
    else:
        p_nie *= prawdopodobienstwa[kolumna][3]  # P(atrybut=nie|spam=nie)

# Normalizacja prawdopodobieństw
suma = p_tak + p_nie
p_tak_norm = p_tak / suma
p_nie_norm = p_nie / suma

# Tworzenie DataFrame z wynikami klasyfikacji
wyniki_df = pd.DataFrame({
    "P(SPAM)": [round(p_tak_norm, 4)],
    "P(NIE-SPAM)": [round(p_nie_norm, 4)],
    "Klasyfikacja": ['SPAM' if p_tak_norm > p_nie_norm else 'NIE-SPAM']
})

print(f"\nWyniki klasyfikacji dla wiadomości:\n{pd.DataFrame(ostatni_wiersz).T}\n")
print(wyniki_df)
