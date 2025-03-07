import os
import pandas as pd
import re
import string
import nltk
import time  # Added for timing

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.preprocessing import FunctionTransformer

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from collections import Counter

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ======================
# 1. TEXT PREPROCESSOR (OPTIMIZED)
# ======================
class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.punct_table = str.maketrans('', '', string.punctuation)
        
    def clean_text(self, text: str) -> str:
        # Combined regex operations
        text = re.sub(r'http\S+|www\S+|@\w+|#\w+|\d+', '', text.lower())
        text = text.translate(self.punct_table)
        return ' '.join([
            self.lemmatizer.lemmatize(word)
            for word in text.split()
            if word not in self.stop_words
        ])

# ======================
# 2. DATA LOADING (WITH TIMING)
# ======================
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_data(file_name: str, is_train: bool) -> pd.DataFrame:
    """Generic data loader with timing"""
    start = time.time()
    df = pd.read_csv(os.path.join(data_dir, file_name), encoding='utf-8')
    df.rename(columns={'Text': 'text', 'Label': 'target'}, inplace=True)
    print(f"Loaded {file_name} in {time.time()-start:.1f}s")
    return df

# ======================
# 3. MAIN PROCESSING (OPTIMIZED)
# ======================
def main():
    # Load and preprocess data
    start_total = time.time()
    
    df_train = load_data('train_submission.csv', is_train=True)
    df_train = shuffle(df_train, random_state=42)
    
    preprocessor = TextPreprocessor()
    try:
        import swifter
        df_train['cleaned_text'] = df_train['text'].swifter.apply(preprocessor.clean_text)
    except ImportError:
        print("Swifter not found. Falling back to normal Pandas apply (slower).")
        df_train['cleaned_text'] = df_train['text'].apply(preprocessor.clean_text)

    # Class merging
    MIN_SAMPLES = 4
    class_counts = df_train['target'].value_counts()
    rare_classes = class_counts[class_counts < MIN_SAMPLES].index
    df_train['target'] = df_train['target'].where(~df_train['target'].isin(rare_classes), 'other')

    # Fixing missing labels
    df_train.dropna(subset=['target'], inplace=True)

    # Fixing missing text
    df_train['cleaned_text'].fillna('', inplace=True)

    # Ensure no empty text
    df_train = df_train[df_train['cleaned_text'].str.strip() != '']

    # Final debug check
    print("Final check before splitting...")
    print("NaNs in 'cleaned_text':", df_train['cleaned_text'].isna().sum())
    print("NaNs in 'target':", df_train['target'].isna().sum())
        
    # Train-validation split
    X_train, X_valid, y_train, y_valid = train_test_split(
        df_train['cleaned_text'],
        df_train['target'],
        test_size=0.25,
        random_state=42,
        stratify=df_train['target']
    )
    
    # ======================
    # 4. OPTIMIZED MODEL PIPELINE
    # ======================
    final_classes = y_train.nunique()
    
    if final_classes > 1:
        # Faster TF-IDF with limited features
        tfidf = TfidfVectorizer(max_features=300, min_df=2, max_df=0.9)
        
        # Optimized RandomForest with parallelization
        rf = RandomForestClassifier(
            class_weight='balanced',
            n_estimators=50,  # Reduced from 100
            max_depth=10,    # Added depth limit
            n_jobs=-1,        # Parallel processing
            random_state=42
        )
        
        
        # SMOTE configuration
        class_counts = Counter(y_train)
        min_train_count = min(class_counts.values())
        smote_k_neighbors = min(1, min_train_count - 1)
        
        pipeline = Pipeline([
            ('tfidf', tfidf),
            ('to_dense', FunctionTransformer(lambda x: x.toarray())),
            ('smote', SMOTE(
                random_state=42,
                k_neighbors=smote_k_neighbors,
                sampling_strategy='minority'  # Limit oversampling
            )),
            ('rf', rf)
        ])
        
        # Faster hyperparameter search
        search = RandomizedSearchCV(
            pipeline,
            {
                'rf__n_estimators': [30, 50],  # Reduced options
                'tfidf__max_features': [200, 300],
                'tfidf__ngram_range': [(1,1), (1,2)]
            },
            n_iter=2,
            cv=2,
            scoring='f1_weighted',
            n_jobs=-1,  # Parallelize search
            verbose=1,
            random_state=42
        )
        
        # Train with timing
        print("Starting training...")
        start_train = time.time()
        search.fit(X_train, y_train)
        print(f"Training completed in {time.time()-start_train:.1f}s")
        
        model = search.best_estimator_
        print("Best params:", search.best_params_)
        
        # Validation
        if len(X_valid) > 0:
            start_pred = time.time()
            y_pred = model.predict(X_valid)
            print(f"Prediction time: {time.time()-start_pred:.1f}s")
            print(classification_report(y_valid, y_pred))
    
    # ======================
    # 5. PREDICT & SAVE
    # ======================
    df_test = load_data('test_without_labels.csv', is_train=False)
    if not df_test.empty:
        df_test['cleaned_text'] = df_test['text'].apply(preprocessor.clean_text)
        submission = pd.DataFrame({
            'ID': df_test.index,
            'Usage': df_test['Usage'],
            'Label': model.predict(df_test['cleaned_text'])
        })
        submission.to_csv(os.path.join(data_dir, 'submission.csv'), index=False)
    
    print(f"Total execution time: {time.time()-start_total:.1f}s")

if __name__ == "__main__":
    main()