import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Klasyfikator Bayes'a
model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(X_test)

# Wyniki
print("Klasyfikator Bayesa:")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
print("F1 score: ", f1_score(y_test, y_pred, average="weighted"))

print(y_test)
print(y_pred)

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
print("\nPrzewidywany wynik (czy zagrają mecz):", predicted_label[0])


# k-NN (odległość euklidesowa)
knn_euclidean = KNeighborsClassifier(metric='euclidean', n_neighbors=2)
knn_euclidean.fit(X_train, y_train)
y_pred_knn_euclidean = knn_euclidean.predict(X_test)

# Wyniki dla odległości euklidesowej
print("k-NN z odległością euklidesową:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn_euclidean)}")
print(classification_report(y_test, y_pred_knn_euclidean))


# k-NN (odległość miejska)
knn_manhattan = KNeighborsClassifier(metric='manhattan', n_neighbors=2)
knn_manhattan.fit(X_train, y_train)
y_pred_knn_manhattan = knn_manhattan.predict(X_test)

# Wyniki dla odległości miejskiej
print("k-NN z odległością miejską:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn_manhattan)}")
print(classification_report(y_test, y_pred_knn_manhattan))
