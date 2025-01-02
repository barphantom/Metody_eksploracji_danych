import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Wczytanie danych
dane = pd.read_csv("Lab_2-zad_4.csv", delimiter=";")

# Model 1 - Zależność kalorii względem tłuszczy zawartych w produkcie
dane["tluszcz"] = dane["tluszcz"]>1
dane["tluszcz"] = dane["tluszcz"].astype("category").cat.codes
x1 = np.array(dane["kalorie"]).reshape(-1, 1)
y1 = np.array(dane["tluszcz"])
model1 = LogisticRegression()
model1.fit(x1, y1)
sns.regplot(data=dane, x=x1, y=y1, marker="x", color=".3", line_kws=dict(color="r"), logistic=True, ci=20)
print(f"Dopasowanie modelu 1: {model1.score(x1, y1)}")
plt.xlabel("Kalorie")
plt.ylabel("Tłuszcze")
plt.title("Zależność kalorii względem tłuszczy zawartych w produkcie")
plt.grid(alpha=0.4, linestyle="--", linewidth=0.5, color="gray")
plt.savefig("Lab_2-zad_4-model_1.png", dpi=300)
plt.show()

# Model 2 - Zależność ilości cukru od występowania na półce nr 1
x2 = np.array(dane["cukry"]).reshape(-1, 1)
y2 = np.array(dane["polka_1"])
model2 = LogisticRegression()
model2.fit(x2, y2)
sns.regplot(data=dane, x=x2, y=y2, marker="x", color=".3", line_kws=dict(color="r"), logistic=True, ci=20)
print(f"Dopasowanie modelu 2: {model2.score(x2, y2)}")
plt.xlabel("Cukry")
plt.ylabel("Półka 1")
plt.title("Zależność ilości cukru od występowania na półce nr 1")
plt.grid(alpha=0.4, linestyle="--", linewidth=0.5, color="gray")
plt.savefig("Lab_2-zad_4-model_2.png", dpi=300)
plt.show()

# Model 3 - Zależność występowania cukru w składzie w zależności od ilości potasu
dane["potas"] = dane["potas"]>180
dane["potas"] = dane["potas"].astype("category").cat.codes
x3 = np.array(dane["cukry"]).reshape(-1, 1)
y3 = np.array(dane["potas"])
model3 = LogisticRegression()
model3.fit(x3, y3)
sns.regplot(data=dane, x=x3, y=y3, marker="x", color=".3", line_kws=dict(color="r"), logistic=True, ci=20)
print(f"Dopasowanie modelu 3: {model3.score(x3, y3)}")
plt.xlabel("Tłuszcze")
plt.ylabel("Sód")
plt.title("Zależność występowania cukru w składzie w zależności od ilości potasu")
plt.grid(alpha=0.4, linestyle="--", linewidth=0.5, color="gray")
plt.savefig("Lab_2-zad_4-model_3.png", dpi=300)
plt.show()