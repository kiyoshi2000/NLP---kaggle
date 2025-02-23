import pandas as pd
import os
import nltk
import re
import string
import time
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Download necessary resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Set data directory
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

# Function to load data safely
def load_data(file_name: str) -> pd.DataFrame:
    start = time.time()
    df = pd.read_csv(os.path.join(data_dir, file_name), encoding='utf-8', low_memory=False)
    print(f"Loaded {file_name} in {time.time()-start:.1f}s")
    return df

# Initialize lemmatizer and POS tag conversion
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    """Map POS tag to first character used by WordNetLemmatizer"""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default to noun

# Enhanced text preprocessing
def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return "missing_text"

    # Text normalization
    text = text.lower()
    text = re.sub(r'http\S+', ' URL ', text)          # Replace URLs
    text = re.sub(r'@\w+', ' MENTION ', text)         # Replace mentions
    text = re.sub(r'#(\w+)', r' HASHTAG_\1 ', text)   # Preserve hashtag content
    text = re.sub(r'\d+', ' NUM ', text)              # Replace numbers
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)  # Handle punctuation
    text = re.sub(r'\s+', ' ', text).strip()          # Remove extra spaces

    # Tokenization and POS tagging
    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    
    # POS-aware lemmatization
    lemmatized_tokens = [
        lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag))
        for word, tag in pos_tags
    ]
    
    return " ".join(lemmatized_tokens) if lemmatized_tokens else "missing_text"

# Load and prepare data
train_df = load_data('train_submission.csv')

# Drop missing values in essential columns
train_df.dropna(subset=['Text', 'Label'], inplace=True)

# **Apply preprocessing BEFORE train-test split**
start_preprocess = time.time()
train_df['Processed_Text'] = train_df['Text'].apply(preprocess_text)
print(f"Preprocessing time: {time.time() - start_preprocess:.2f} seconds")

# Filter out classes with less than 2 samples before splitting
min_samples = 2
label_counts = train_df['Label'].value_counts()
valid_labels = label_counts[label_counts >= min_samples].index

print(f"Original classes: {len(label_counts)}")
print(f"Classes with at least {min_samples} samples: {len(valid_labels)}")

# Filter the dataframe
train_df = train_df[train_df['Label'].isin(valid_labels)]

# Convert labels to string for classification
train_df['Label'] = train_df['Label'].astype(str)

# Verify new distribution
print("\nFiltered class distribution:")
print(train_df['Label'].value_counts())

# **Now perform train-test split**
X = train_df['Processed_Text']  # ✅ Now it exists
y = train_df['Label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Optimized pipeline with SVM
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(
        max_features=15000,
        min_df=3,
        max_df=0.75,
        ngram_range=(1, 3),
        sublinear_tf=True
    )),
    ('classifier', LinearSVC(
        class_weight='balanced',
        C=0.5,
        max_iter=10000,
        random_state=42
    ))
])

# Train model
start_time = time.time()
pipeline.fit(X_train, y_train)
print(f"Training time: {time.time() - start_time:.2f} seconds")

# Evaluate model
y_pred = pipeline.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Generate predictions for test data
test_df = load_data('test_without_labels.csv')

# **Preprocess the test data**
test_df['Processed_Text'] = test_df['Text'].apply(preprocess_text)
test_df['Predicted_Label'] = pipeline.predict(test_df['Processed_Text'])

# Save results
test_df['ID'] = range(1, len(test_df) + 1)
output_path = os.path.join(data_dir, 'test_predictions.csv')
test_df[['ID', 'Predicted_Label']].rename(columns={'Predicted_Label': 'Label'}).to_csv(output_path, index=False)
print(f"Predictions saved to: {output_path}")
