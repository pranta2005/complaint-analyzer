"""
CampusVoice Complaint Analyzer — Model Evaluation (Improved)
Uses 5-fold Stratified Cross-Validation for reliable metrics.
"""
import pandas as pd
import re
import os
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.metrics import precision_score, recall_score, f1_score

# ── Label Maps (same as train.py) ─────────────────────────────────────────────
CATEGORY_MAP = {
    'Academic':     'Academics',
    'Billing':      'Finance',
    'Office':       'Administration',
    'HR':           'Administration',
    'Events':       'Extracurricular',
    'Cloud':        'Software',
    'Database':     'Software',
    'Access':       'Software',
    'Health':       'Facilities',
    'Environment':  None,
    'Mobile':       None,
}
PRIORITY_MAP = {'Critical': 'High'}


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_clean(csv_path: str, min_words: int = 6) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep='\t', on_bad_lines='skip')
    df.fillna('', inplace=True)
    df['category'] = df['category'].map(lambda x: CATEGORY_MAP.get(x, x))
    df['priority']  = df['priority'].map(lambda x: PRIORITY_MAP.get(x, x))
    df = df[df['category'].notna()].reset_index(drop=True)
    df['text'] = df['text'].apply(clean_text)
    # Only keep complaints with enough words for reliable classification
    df = df[df['text'].str.split().str.len() >= min_words].reset_index(drop=True)
    return df


script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path   = os.path.join(script_dir, 'complaints_augmented.csv')
df = load_and_clean(csv_path)

print(f"Dataset: {len(df)} complaints | {df['category'].nunique()} categories | {df['priority'].nunique()} priority levels")
print(f"Category distribution:\n{df['category'].value_counts().to_string()}\n")

# ── Pipeline factory ───────────────────────────────────────────────────────────
def build_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            stop_words='english',
        )),
        ('clf', CalibratedClassifierCV(
            LinearSVC(max_iter=2000, class_weight='balanced', C=1.0), cv=3
        )),
    ])


# ── Scorers ────────────────────────────────────────────────────────────────────
scorers = {
    'accuracy':  'accuracy',
    'precision': make_scorer(precision_score, average='weighted', zero_division=0),
    'recall':    make_scorer(recall_score,    average='weighted', zero_division=0),
    'f1':        make_scorer(f1_score,        average='weighted', zero_division=0),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def evaluate(label: str, target_col: str):
    print("=" * 65)
    print(f"MODEL: {label}")
    print(f"  Algorithm : LinearSVC  (calibrated for probabilities)")
    print(f"  Features  : TF-IDF bigrams, 10k features, sublinear_tf=True")
    print(f"  Evaluation: 5-Fold Stratified Cross-Validation")
    print("=" * 65)

    pipe = build_pipeline()
    results = cross_validate(pipe, df['text'], df[target_col], cv=cv, scoring=scorers, n_jobs=-1)

    acc  = results['test_accuracy'].mean()
    prec = results['test_precision'].mean()
    rec  = results['test_recall'].mean()
    f1   = results['test_f1'].mean()

    print(f"\n  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")

    # Per-class report using last fold
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        df['text'], df[target_col], test_size=0.2, random_state=42, stratify=df[target_col]
    )
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    print(f"\n  Per-Class Report (held-out 20% split):")
    print(classification_report(y_te, y_pred, zero_division=0))

    return acc, prec, rec, f1


cat_acc,  cat_prec,  cat_rec,  cat_f1  = evaluate("Category Classifier", "category")
pri_acc,  pri_prec,  pri_rec,  pri_f1  = evaluate("Priority Classifier",  "priority")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("FINAL SUMMARY  (5-Fold Cross-Validation — weighted averages)")
print("=" * 75)
print(f"{'Model':<32} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print("-" * 75)
print(f"{'Category Classifier (LinearSVC)':<32} {cat_acc:>10.4f} {cat_prec:>10.4f} {cat_rec:>10.4f} {cat_f1:>10.4f}")
print(f"{'Priority Classifier (LinearSVC)':<32} {pri_acc:>10.4f} {pri_prec:>10.4f} {pri_rec:>10.4f} {pri_f1:>10.4f}")
print("=" * 75)
