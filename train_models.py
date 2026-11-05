import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 1. Load and Preprocess Data (matching your clean pipeline)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

df = pd.read_csv('clickbait_data.csv')

df['headline_lower'] = df['headline'].str.lower()


def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


df['cleaned_headline'] = df['headline_lower'].apply(clean_text)

stop_words = set(stopwords.words('english'))


def tokenize_and_remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)


df['processed_headline'] = df['cleaned_headline'].apply(tokenize_and_remove_stopwords)

# 2. Vectorization & Feature Target Split
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['processed_headline'])
y = df['clickbait']

# 3. Split into 70% Training Set and 30% Test Set[cite: 2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}\n")

# 4. Train and Evaluate Models[cite: 2]
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Multinomial Naïve Bayes": MultinomialNB(),
    "Support Vector Machine (SVM)": SVC()
}

for name, model in models.items():
    print(f"--- Training {name} ---")
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics[cite: 2]
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}\n")