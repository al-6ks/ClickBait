import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# 1. Load the dataset
print("Loading dataset...")
df = pd.read_csv('clickbait_data.csv')

# Verify distribution from your project specs (32,000 total rows, no missing values)[cite: 2]
print("\n--- Dataset Info ---")
print(df.head())
print("\nClass Distribution:")
print(df['clickbait'].value_counts())
print("Missing values:", df.isnull().sum().to_dict())

# 2. Text Preprocessing & Cleaning
print("\nCleaning text...")
df['headline_lower'] = df['headline'].str.lower()

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and excessive punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

df['cleaned_headline'] = df['headline_lower'].apply(clean_text)

# Tokenize and remove stop words
stop_words = set(stopwords.words('english'))

def tokenize_and_remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

df['processed_headline'] = df['cleaned_headline'].apply(tokenize_and_remove_stopwords)

# 3. Structural Feature Engineering
df['headline_length'] = df['headline'].apply(len)
df['num_question_marks'] = df['headline'].apply(lambda x: x.count('?'))
df['num_exclamation_marks'] = df['headline'].apply(lambda x: x.count('!'))
df['num_numerals'] = df['headline'].apply(lambda x: sum(c.isdigit() for c in x))

# 4. TF-IDF Vectorization
print("Applying TF-IDF vectorization...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(df['processed_headline'])

print("\nPreprocessing complete! TF-IDF Feature Matrix Shape:", X_tfidf.shape)