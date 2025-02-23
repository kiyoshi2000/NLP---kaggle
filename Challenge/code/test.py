import pandas as pd
import os
import nltk
import re
import string
import time
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Download necessary resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Set data directory
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

# Function to load data safely
def load_data(file_name: str) -> pd.DataFrame:
    start = time.time()
    df = pd.read_csv(os.path.join(data_dir, file_name), encoding='utf-8', low_memory=False)
    print(f"Loaded {file_name} in {time.time()-start:.1f}s")
    return df

# Initialize stopwords and lemmatizer globally
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Optimized text preprocessing function
def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return "missing_text"  # Replace empty values with a placeholder

    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return " ".join(tokens) if tokens else "missing_text"

# Load training data
train_df = load_data('train_submission.csv')

# Ensure there are no missing values in `Text` or `Label`
train_df.dropna(subset=['Text', 'Label'], inplace=True)

# Apply preprocessing
start_preprocess = time.time()
train_df['Processed_Text'] = train_df['Text'].apply(preprocess_text)
print(f"Preprocessing time: {time.time() - start_preprocess:.2f} seconds")

# Convert `Label` to string to avoid numerical NaN issues
train_df['Label'] = train_df['Label'].astype(str)

# Split dataset into train/test
X = train_df['Processed_Text']
y = train_df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure there are no NaNs after splitting
X_train = X_train.dropna()
y_train = y_train.dropna()

# Convert y_train to a NumPy array
y_train = np.array(y_train)

# Build optimized pipeline
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(max_features=5000, sublinear_tf=True, ngram_range=(1,2))),
    ('classifier', RandomForestClassifier(n_estimators=50, max_depth=30, min_samples_split=5, random_state=42, n_jobs=-1, verbose=1))
])

# Train model
start_time = time.time()
pipeline.fit(X_train, y_train)
print(f"Training time: {time.time() - start_time:.2f} seconds")

# Evaluate model
y_pred = pipeline.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Load test dataset (without labels)
test_df = load_data('test_without_labels.csv')

# Apply preprocessing to test data
start_preprocess_test = time.time()
test_df['Processed_Text'] = test_df['Text'].apply(preprocess_text)
print(f"Test preprocessing time: {time.time() - start_preprocess_test:.2f} seconds")

# Predict labels for test set
test_df['Predicted_Label'] = pipeline.predict(test_df['Processed_Text'])

# Add ID column (index of rows as unique ID)
test_df['ID'] = range(1, len(test_df) + 1)

# Define the output path inside the 'data' folder
output_path = os.path.join(data_dir, 'test_predictions.csv')

# Save results with only two columns: ID and Label inside the 'data' folder
test_df[['ID', 'Predicted_Label']].rename(columns={'Predicted_Label': 'Label'}).to_csv(output_path, index=False)

print(f"Predictions saved to: {output_path}")
