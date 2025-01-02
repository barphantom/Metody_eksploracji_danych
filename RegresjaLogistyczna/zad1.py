import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
import seaborn as sns

# 1. Przygotowanie danych
# Dane na podstawie tabeli (rok studiów i stan cywilny)
data = {
    "rokStudiow": [1, 2, 5, 1, 4, 3, 2, 1, 5, 2, 3, 4, 1, 2, 5, 4, 3, 1, 4, 5, 2, 5, 3, 4, 3, 2, 5, 1],
    "stanCywilny": [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0]
}
dff = pd.DataFrame(data)
df = dff.sort_values(by="rokStudiow")

print(df)
# print(sorted_df)
X = df["rokStudiow"].values.reshape(-1, 1)
print(X)
y = df["stanCywilny"].values.reshape(-1, 1)
print(y)

# 2. Liniowy Model Prawdopodobieństwa (LPM) przy użyciu LinearRegression
# X = df["rokStudiow"].values.reshape(-1, 1)
# y = df["stanCywilny"].values.reshape(-1)
X = np.array(df["rokStudiow"]).reshape(-1, 1)
y = np.array(df["stanCywilny"]).reshape(-1)


linear_model = LinearRegression()
linear_model.fit(X, y)
predictions_lpm = linear_model.predict(X)

# 3. Model Logitowy przy użyciu LogisticRegression
logit_model = LogisticRegression()
logit_model.fit(X, y)
predictions_logit = logit_model.predict(X)
predictions_logit_prob = logit_model.predict_proba(X)[:, 1]  # Prawdopodobieństwa dla klasy 1

# 4. Wizualizacja wyników
plt.figure(figsize=(10, 6))
plt.scatter(df["rokStudiow"], df["stanCywilny"], color="blue", label="Dane rzeczywiste", alpha=0.7)
# plt.plot(X, predictions_lpm, color="green", label="LPM - prognoza", linewidth=2)
plt.plot(X, predictions_logit_prob, color="red", label="Logit - prognoza", linewidth=2)
# plt.scatter(X, predictions_logit_prob, color="red", marker="+")
# sns.regplot(x=X, y=y, data=df, logistic=True, ci=None)

plt.xlabel("Rok Studiów")
plt.ylabel("Stan Cywilny (0 = W, 1 = M)")
plt.title("Porównanie modeli LPM i Logitowego")
plt.legend()
plt.grid(True)
plt.show()
