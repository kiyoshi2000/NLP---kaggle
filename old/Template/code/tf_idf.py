import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm

import re
import string

import nltk
from sklearn.utils import shuffle
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
stop_words = stopwords.words('english')

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns
import os

# Data
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

file_path_test = os.path.join(data_dir, 'test.json')
file_path_train = os.path.join(data_dir, 'train.json')

with open(file_path_train, 'r', encoding='utf-8') as file:
    train_data = json.load(file)

with open(file_path_test, 'r', encoding='utf-8') as file:
    test_data = json.load(file)

# Process training data
train_processed = {'text': [], 'target': []}
for label, samples in train_data.items():
    train_processed['text'] += samples
    train_processed['target'] += [label] * len(samples)

df_train = pd.DataFrame.from_dict(train_processed)

#Process test data (no labels)
test_processed = {'text': []}
for _, samples in test_data.items():
    test_processed['text'] += samples

df_test = pd.DataFrame.from_dict(test_processed)

# Train-validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    df_train['text'].values, df_train['target'].values, test_size=0.2, random_state=123
)

# Shuffle training data
X_train, y_train = shuffle(X_train, y_train, random_state=123)

# Save shuffled training data (optional)
train_shuffle_path = os.path.join(data_dir, 'train_shuffle.txt')
y_train_shuffle_path = os.path.join(data_dir, 'y_train_shuffle.txt')

with open(train_shuffle_path, 'w', encoding='utf-8') as file:
    for item in X_train.tolist():
        file.write(item + "\n")

with open(y_train_shuffle_path, 'w', encoding='utf-8') as file:
    for item in y_train.tolist():
        file.write(item + "\n")

# Initialize and fit TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_train_vectors = tfidf_vectorizer.fit_transform(X_train)
tfidf_valid_vectors = tfidf_vectorizer.transform(X_valid)
tfidf_test_vectors = tfidf_vectorizer.transform(df_test['text'].values)

# Train classifier
classifier = RandomForestClassifier()
classifier.fit(tfidf_train_vectors, y_train)

# Evaluate model on validation set
y_valid_pred = classifier.predict(tfidf_valid_vectors)
y_pred_test_path = os.path.join(data_dir, 'y_pred_test.txt')


print("Validation Performance:")
print(classification_report(y_valid, y_valid_pred))

# Make predictions on the test set
y_test_pred = classifier.predict(tfidf_test_vectors)

# Save test set predictions
with open(y_pred_test_path, 'w', encoding='utf-8') as file:
    for item in y_test_pred.tolist():
        file.write(item + "\n")