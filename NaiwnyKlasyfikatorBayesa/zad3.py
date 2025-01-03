import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Pobierz wymagane zasoby NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Funkcja do przetwarzania tekstu
def preprocess_text(text):
    # Tokenizacja
    tokens = word_tokenize(text.lower())
    # Usuwanie stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    # Lematyzacja
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Wczytaj dane z pliku Excel
file_path = "./MED-lab-3-Zad 3-Mandrill-Dane.xlsx"  # Zmień na swoją ścieżkę
mandrill_posts = pd.read_excel(file_path, sheet_name='dot. aplikacji Mandrill')
other_posts = pd.read_excel(file_path, sheet_name='dot. innych')

# Dodaj etykiety
mandrill_posts['label'] = 'Mandrill'
other_posts['label'] = 'inne'

# Połącz dane
data = pd.concat([mandrill_posts, other_posts], ignore_index=True)
data['Post'] = data['Post'].astype(str)

# Przetwórz teksty
data['Processed_Post'] = data['Post'].apply(preprocess_text)

# Podział na dane i etykiety
X = data['Processed_Post']
y = data['label']

# Podział na zestawy treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Naiwny klasyfikator Bayesowski
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predykcja
y_pred = model.predict(X_test_tfidf)

# Wyniki
print("Dokładność:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Przykład klasyfikacji nowego posta
new_post = ["Mandrill is a fantastic tool for developers."]
new_post_processed = preprocess_text(new_post[0])
new_post_tfidf = vectorizer.transform([new_post_processed])
prediction = model.predict(new_post_tfidf)
print(f"Predykcja dla nowego posta: {prediction[0]}")

print("\n")
for sym in range(len(y_pred)):
    print(f"Post: {X_test.iloc[sym]} | Etykieta rzeczywista: {y_test.iloc[sym]} | Etykieta przewidziana: {y_pred[sym]}")

