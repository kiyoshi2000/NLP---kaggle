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
from imblearn.over_sampling import SMOTE, RandomOverSampler

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

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# POS tag conversion for lemmatization
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Text preprocessing function
def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return "missing_text"

    text = text.lower()
    text = re.sub(r'http\S+', ' URL ', text)
    text = re.sub(r'@\w+', ' MENTION ', text)
    text = re.sub(r'#(\w+)', r' HASHTAG_\1 ', text)
    text = re.sub(r'\d+', ' NUM ', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text, language="english")
    pos_tags = nltk.pos_tag(tokens)

    lemmatized_tokens = [
        lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag))
        for word, tag in pos_tags
    ]
    
    return " ".join(lemmatized_tokens) if lemmatized_tokens else "missing_text"

# Load and preprocess data
train_df = load_data('train_submission.csv')
train_df.dropna(subset=['Text', 'Label'], inplace=True)

start_preprocess = time.time()
train_df['Processed_Text'] = train_df['Text'].apply(preprocess_text)
print(f"Preprocessing time: {time.time() - start_preprocess:.2f} seconds")

# Check class distribution before balancing
label_counts = train_df['Label'].value_counts()
print("\nClass distribution before balancing:")
print(label_counts)

# Separate classes with only 1 sample
single_sample_classes = label_counts[label_counts == 1].index

# Duplicate single-sample classes to allow for stratified splitting
for cls in single_sample_classes:
    train_df = pd.concat([train_df, train_df[train_df['Label'] == cls]], ignore_index=True)

# Train-test split (now works without removing single-instance classes)
X = train_df['Processed_Text']
y = train_df['Label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(
    max_features=15000,
    min_df=3,
    max_df=0.75,
    ngram_range=(1, 3),
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Determine class balancing strategy
class_counts = pd.Series(y_train).value_counts()
dominant_class_threshold = class_counts.max() * 0.3  # Classes with more than 30% of dominant class are ignored

# Define which classes should be oversampled with SMOTE
smote_classes = class_counts[(class_counts >= 6) & (class_counts < dominant_class_threshold)].index
random_oversample_classes = class_counts[class_counts < 6].index  # Now includes 1-sample classes

# Apply SMOTE only to medium-sized underrepresented classes
if len(smote_classes) > 0:
    smote = SMOTE(sampling_strategy={cls: min(class_counts.max(), int(class_counts[cls] * 2))
                                     for cls in smote_classes},
                  k_neighbors=3, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vec, y_train)
else:
    X_train_resampled, y_train_resampled = X_train_vec, y_train  # No SMOTE applied if conditions not met

# Apply Random Oversampling for very small classes (now includes those with 1-5 samples)
if len(random_oversample_classes) > 0:
    random_oversampler = RandomOverSampler(sampling_strategy={cls: 6 for cls in random_oversample_classes},
                                           random_state=42)
    X_train_resampled, y_train_resampled = random_oversampler.fit_resample(X_train_resampled, y_train_resampled)

# Check new class distribution after balancing
print("\nClass distribution after balancing:")
print(pd.Series(y_train_resampled).value_counts())

# Define the classifier pipeline
classifier = Pipeline([
    ('classifier', LinearSVC(
        class_weight=None,
        C=0.5,
        max_iter=10000,
        random_state=42
    ))
])

# Train the model
start_time = time.time()
classifier.fit(X_train_resampled, y_train_resampled)
print(f"Training time: {time.time() - start_time:.2f} seconds")

# Evaluate the model
y_pred = classifier.predict(X_test_vec)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Load test data and generate predictions
test_df = load_data('test_without_labels.csv')
test_df['Processed_Text'] = test_df['Text'].apply(preprocess_text)

test_vec = vectorizer.transform(test_df['Processed_Text'])
test_df['Predicted_Label'] = classifier.predict(test_vec)

# Save predictions
test_df['ID'] = range(1, len(test_df) + 1)
output_path = os.path.join(data_dir, 'test_predictions.csv')
test_df[['ID', 'Predicted_Label']].rename(columns={'Predicted_Label': 'Label'}).to_csv(output_path, index=False)
print(f"Predictions saved to: {output_path}")
