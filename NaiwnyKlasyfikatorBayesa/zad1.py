import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_excel("./daneZad1.xlsx")
print(df)

# Przekształcenie danych na numeryczne
label_encoders = {}
for column in ['Siła wiatru', 'Zachmurzenie', 'Odczuwalna temperatura', 'Zagrano mecz']:
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])
    label_encoders[column] = label_encoder

X = df[['Siła wiatru', 'Zachmurzenie', 'Odczuwalna temperatura']]
y = df['Zagrano mecz']
print(X)
print(y)

# Klasyfikator Bayes'a
model = GaussianNB()
model.fit(X, y)

# Wynik dla zapytania z pdf'a
new_data = {
    'Siła wiatru': ['silny'],
    'Zachmurzenie': ['słonecznie'],
    'Odczuwalna temperatura': ['ciepło']
}

for column in new_data:
    new_data[column] = label_encoders[column].transform(new_data[column])

new_df = pd.DataFrame(new_data)

prediction = model.predict(new_df)
predicted_label = label_encoders['Zagrano mecz'].inverse_transform(prediction)

print("\nDane wejściowe:")
print(new_df)
print("\nNaiwny klasyfikator Bayes'a:")
print("Przewidywany wynik (czy zagrają mecz):", predicted_label[0])


# k-NN (odległość euklidesowa)
knn_euclidean = KNeighborsClassifier(metric='euclidean', n_neighbors=6)
knn_euclidean.fit(X, y)
y_pred_knn_euclidean = knn_euclidean.predict(new_df)
euk_predicted = label_encoders['Zagrano mecz'].inverse_transform(y_pred_knn_euclidean)

# Wyniki dla odległości euklidesowej
print("\nk-NN z odległością euklidesową:")
print("Przewidywany wynik (czy zagrają mecz):", euk_predicted[0])


# k-NN (odległość miejska)
knn_manhattan = KNeighborsClassifier(metric='manhattan', n_neighbors=6)
knn_manhattan.fit(X, y)
y_pred_knn_manhattan = knn_manhattan.predict(new_df)
manh_predicted = label_encoders['Zagrano mecz'].inverse_transform(y_pred_knn_manhattan)

# Wyniki dla odległości miejskiej
print("\nk-NN z odległością miejską:")
print("Przewidywany wynik (czy zagrają mecz):", manh_predicted[0])

