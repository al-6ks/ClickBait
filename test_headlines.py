import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Ensure NLTK resources are available
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# 1. Load dataset and preprocess
print("Loading dataset and preprocessing...")
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

# 2. Vectorize and Split Data (using 70% train / 30% test)[cite: 2]
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['processed_headline'])
y = df['clickbait']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Model trained successfully!\n")

# 3. Evaluate Test Set, Confusion Matrix & Metrics
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Calculate Specificity (True Negative Rate)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

print("=" * 50)
print("MODEL EVALUATION & COMPREHENSIVE METRICS (Test Set)")
print("=" * 50)
print(f"Accuracy:    {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision:   {precision_score(y_test, y_pred):.4f}")
print(f"Recall:      {recall_score(y_test, y_pred):.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1-Score:    {f1_score(y_test, y_pred):.4f}\n")

print("Confusion Matrix Breakdown:")
print(f"  * True Negatives (Correct Non-Clickbait): {tn}")
print(f"  * False Positives (Legitimate flagged as Clickbait): {fp}")
print(f"  * False Negatives (Missed Clickbait): {fn}")
print(f"  * True Positives (Correct Clickbait): {tp}\n")

# 4. Test on New, Unseen Headlines
new_headlines = [
    "You won't believe what this celebrity did on live TV!",
    "Scientists discover a new exoplanet 40 light-years away.",
    "10 shocking secrets doctors don't want you to know about weight loss!",
    "Local university releases admission guidelines for the upcoming academic year."
]

cleaned_new = [clean_text(h.lower()) for h in new_headlines]
processed_new = [tokenize_and_process(h) if 'tokenize_and_process' in locals() else tokenize_and_remove_stopwords(h) for h in cleaned_new]
X_new = vectorizer.transform(processed_new)

predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)

print("=" * 50)
print("RESULTS FOR UNSEEN HEADLINES TEST")
print("=" * 50)
for headline, pred, prob in zip(new_headlines, predictions, probabilities):
    label = "Clickbait (1)" if pred == 1 else "Non-Clickbait (0)"
    confidence = prob[pred] * 100
    print(f"Headline:   '{headline}'")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2f}%\n")