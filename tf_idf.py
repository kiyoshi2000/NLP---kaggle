import pandas as pd
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE

# 1. Download NLTK data if needed
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ======================
# 2. TEXT PREPROCESSOR
# ======================
class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        # Table to remove punctuation
        self.punct_table = str.maketrans('', '', string.punctuation)
        
    def clean_text(self, text: str) -> str:
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
        # Lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(self.punct_table)
        # Remove digits
        text = re.sub(r'\d+', '', text)
        # Tokenize
        tokens = text.split()
        # Lemmatize and remove stopwords
        tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in self.stop_words
        ]
        return ' '.join(tokens)

# =========================
# 3. FUNCTIONS TO LOAD DATA
# =========================
def load_train_csv(file_path: str) -> pd.DataFrame:
    """
    Expects a CSV with columns: Usage, Text, Label.
    Returns a DataFrame with columns: [Usage, text, target].
    """
    df = pd.read_csv(file_path, encoding='utf-8', sep=',')
    df.rename(columns={'Text': 'text', 'Label': 'target'}, inplace=True)
    return df

def load_test_csv(file_path: str) -> pd.DataFrame:
    """
    Expects a CSV with columns: Usage, Text.
    Returns a DataFrame with columns: [Usage, text].
    """
    df = pd.read_csv(file_path, encoding='utf-8', sep=',')
    df.rename(columns={'Text': 'text'}, inplace=True)
    return df

# =====================
# 4. LOAD TRAINING DATA
# =====================
df_train = load_train_csv('train_submission.csv')  # or .txt if needed

# Optional: Shuffle the training data
df_train = shuffle(df_train, random_state=42).reset_index(drop=True)

# Preprocess the text column
preprocessor = TextPreprocessor()
df_train['cleaned_text'] = df_train['text'].apply(preprocessor.clean_text)

# ==================================
# 5. MERGE RARE CLASSES INTO "other"
# ==================================
MIN_SAMPLES = 2  # or whichever threshold you prefer
class_counts = df_train['target'].value_counts()
rare_classes = class_counts[class_counts < MIN_SAMPLES].index

if len(rare_classes) > 0:
    print(f"Merging {len(rare_classes)} class(es) into 'other': {list(rare_classes)}")
    df_train['target'] = df_train['target'].apply(
        lambda x: 'other' if x in rare_classes else x
    )

# =====================================================
# 6. DROP ROWS WITH MISSING LABELS OR MISSING TEXT (NaN)
# =====================================================
print("Missing in 'target':", df_train['target'].isna().sum())
print("Missing in 'cleaned_text':", df_train['cleaned_text'].isna().sum())

df_train.dropna(subset=['target', 'cleaned_text'], how='any', inplace=True)

# Check final class distribution
print("Final class distribution:")
print(df_train['target'].value_counts())

final_classes = df_train['target'].nunique()
print("Number of final classes:", final_classes)

# ============================
# 7. TRAIN-VALIDATION SPLIT
# ============================
test_size_fraction = 0.25

if final_classes > 1:
    # Attempt stratification if feasible
    try:
        X_train, X_valid, y_train, y_valid = train_test_split(
            df_train['cleaned_text'],
            df_train['target'],
            test_size=test_size_fraction,
            random_state=42,
            stratify=df_train['target']
        )
    except ValueError as e:
        # In case stratification fails, do a normal split
        print("Stratified split failed, using random split. Error:", e)
        X_train, X_valid, y_train, y_valid = train_test_split(
            df_train['cleaned_text'],
            df_train['target'],
            test_size=test_size_fraction,
            random_state=42
        )
else:
    # Only one class => no point in splitting
    X_train = df_train['cleaned_text']
    y_train = df_train['target']
    X_valid, y_valid = [], []

# ====================================================
# 8. DECIDE IF WE USE SMOTE OR NOT (BASED ON CLASS SIZE)
# ====================================================
min_train_count = pd.Series(y_train).value_counts().min()

if final_classes > 1:
    if min_train_count < 2:
        # If any class has just 1 sample, skip SMOTE
        print("WARNING: Some classes have < 2 samples. Skipping SMOTE.")
        pipeline = make_pipeline(
            TfidfVectorizer(max_features=500, min_df=1, max_df=0.95),
            RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
        )
        param_distributions = {
            'randomforestclassifier__n_estimators': [50, 100],
            'tfidfvectorizer__max_features': [300, 500]
        }
    else:
        # Use SMOTE
        print("Using SMOTE with k_neighbors=1.")
        pipeline = make_pipeline(
            TfidfVectorizer(max_features=500, min_df=1, max_df=0.95),
            SMOTE(random_state=42, k_neighbors=1),
            RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
        )
        param_distributions = {
            'randomforestclassifier__n_estimators': [50, 100],
            'tfidfvectorizer__max_features': [300, 500]
        }

    # Perform RandomizedSearchCV for hyperparam tuning
    search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=2,  # keep small for demonstration
        cv=2,
        scoring='f1_weighted',
        verbose=1,
        random_state=42
    )

    search.fit(X_train, y_train)
    model = search.best_estimator_
    print("Best params:", search.best_params_)

    # Evaluate on validation set (if non-empty)
    if len(X_valid) > 0:
        y_valid_pred = model.predict(X_valid)
        print("\nValidation set performance:")
        print(classification_report(y_valid, y_valid_pred))
else:
    # Only one class => dummy classifier
    print("Only one class present. Training a constant classifier.")
    pipeline = make_pipeline(
        TfidfVectorizer(),
        DummyClassifier(strategy='constant', constant=y_train.iloc[0])
    )
    model = pipeline.fit(X_train, y_train)

# ===========================
# 9. LOAD TEST DATA & PREDICT
# ===========================
df_test = load_test_csv('test_without_labels.csv')  # or .txt if needed

if df_test.empty:
    print("Test file is empty. No predictions made.")
else:
    df_test['cleaned_text'] = df_test['text'].apply(preprocessor.clean_text)
    test_pred = model.predict(df_test['cleaned_text'])

    # ============================
    # 10. CREATE SUBMISSION FILE
    # ============================
    # Columns: ID, Usage, Label
    submission_df = pd.DataFrame({
        'ID': df_test.index,
        'Usage': df_test['Usage'],
        'Label': test_pred
    })
    submission_df.to_csv('submission.csv', index=False)
    print("submission.csv generated successfully!")
