"""
CampusVoice Complaint Analyzer — Training Script (Improved)
Models: LinearSVC + TF-IDF (bigrams, sublinear_tf)
Improvements: label normalization, text cleaning, better features
"""
import pandas as pd
import re
import joblib
import os
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# ── Config ────────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path   = os.path.join(script_dir, 'complaints_augmented.csv')

# ── Label Normalization Maps ───────────────────────────────────────────────────
CATEGORY_MAP = {
    'Academic':     'Academics',        # duplicate spelling
    'Billing':      'Finance',          # merge finance-related
    'Office':       'Administration',   # merge admin-related
    'HR':           'Administration',
    'Events':       'Extracurricular',  # merge extra-curricular
    'Cloud':        'Software',         # merge IT-related
    'Database':     'Software',
    'Access':       'Software',
    'Health':       'Facilities',       # merge facilities-related
    'Environment':  None,               # drop (only 2 samples)
    'Mobile':       None,               # drop (only 1 sample)
}

PRIORITY_MAP = {
    'Critical': 'High',  # merge Critical→High (only 12 samples)
}


def clean_text(text: str) -> str:
    """Lowercase, remove special chars, strip whitespace."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_clean(csv_path: str, min_words: int = 6) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep='\t', on_bad_lines='skip')
    df.fillna('', inplace=True)

    # Normalize labels
    df['category'] = df['category'].map(lambda x: CATEGORY_MAP.get(x, x))
    df['priority']  = df['priority'].map(lambda x: PRIORITY_MAP.get(x, x))

    # Drop rows where category was mapped to None
    df = df[df['category'].notna()].reset_index(drop=True)

    # Clean text
    df['text'] = df['text'].apply(clean_text)

    # Filter out very short/ambiguous complaints
    df = df[df['text'].str.split().str.len() >= min_words].reset_index(drop=True)

    return df


# ── Load Data ─────────────────────────────────────────────────────────────────
print("Loading and cleaning dataset...")
df = load_and_clean(csv_path)
print(f"  Loaded {len(df)} complaints across {df['category'].nunique()} categories")
print(f"  Priority classes: {sorted(df['priority'].unique())}\n")

# ── Shared Vectorizer ─────────────────────────────────────────────────────────
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),     # unigrams + bigrams
    sublinear_tf=True,      # log normalization
    min_df=2,               # ignore single-occurrence noise words
    stop_words='english',
)

X = vectorizer.fit_transform(df['text'])

# ── MODEL 1: Category Classifier ─────────────────────────────────────────────
print("Training Category classifier (LinearSVC)...")
svc_cat = LinearSVC(max_iter=2000, class_weight='balanced', C=1.0)
category_clf = CalibratedClassifierCV(svc_cat, cv=3)
category_clf.fit(X, df['category'])

# ── MODEL 2: Priority Classifier ──────────────────────────────────────────────
print("Training Priority classifier (LinearSVC)...")
svc_pri = LinearSVC(max_iter=2000, class_weight='balanced', C=1.0)
priority_clf = CalibratedClassifierCV(svc_pri, cv=3)
priority_clf.fit(X, df['priority'])

# ── Save ──────────────────────────────────────────────────────────────────────
print("\nSaving models to disk...")
joblib.dump(vectorizer,   os.path.join(script_dir, 'vectorizer.pkl'))
joblib.dump(category_clf, os.path.join(script_dir, 'category_model.pkl'))
joblib.dump(priority_clf, os.path.join(script_dir, 'priority_model.pkl'))

print("Training complete! Models saved:")
print(f"  vectorizer.pkl    ({os.path.getsize(os.path.join(script_dir,'vectorizer.pkl'))//1024} KB)")
print(f"  category_model.pkl ({os.path.getsize(os.path.join(script_dir,'category_model.pkl'))//1024} KB)")
print(f"  priority_model.pkl  ({os.path.getsize(os.path.join(script_dir,'priority_model.pkl'))//1024} KB)")
