"""
train.py — Model training, comparison, and persistence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from utils import clean_text, normalise_label


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "sentiment_model.joblib"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.joblib"
LABEL_PATH = MODEL_DIR / "labels.joblib"


def _build_candidates() -> Dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
        ),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Linear SVM": CalibratedClassifierCV(
            LinearSVC(
                max_iter=5000,
                C=0.5,
                class_weight="balanced",
            )
        ),
    }


def train(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    lemmatize: bool = False,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, Any]:

    # Fix duplicate columns
    seen = {}
    new_cols = []

    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}.{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)

    df.columns = new_cols

    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")].reset_index(drop=True)

    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"Selected columns not found. Available columns: {list(df.columns)}"
        )

    # Select required columns
    df = df[[text_col, label_col]].dropna().copy().reset_index(drop=True)

    # Clean text
    df["clean"] = df[text_col].astype(str).apply(
        lambda t: clean_text(t, lemmatize=lemmatize)
    )

    # Normalize labels
    standard = {"Positive", "Negative", "Neutral"}
    unique_vals = set(df[label_col].astype(str).str.strip().unique())

    if unique_vals.issubset(standard):
        df["label"] = df[label_col].astype(str).str.strip()
    else:
        df["label"] = df[label_col].apply(normalise_label)

    df = df[df["label"].isin(standard)]
    df = df[df["clean"].str.strip() != ""].reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid rows found after cleaning labels/text.")

    label_counts = df["label"].value_counts()

    if len(label_counts) < 2:
        raise ValueError("Dataset must contain at least 2 sentiment classes.")

    if label_counts.min() < 2:
        raise ValueError(
            "Each sentiment class must have at least 2 samples for train/test split."
        )

    X = df["clean"].astype(str).to_numpy()
    y = df["label"].astype(str).to_numpy()

    labels = np.array(sorted(set(y)))

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # TF-IDF
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=50000,
        sublinear_tf=True,
        min_df=1,
        max_df=0.95,
        strip_accents="unicode",
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    candidates = _build_candidates()

    all_metrics = {}
    best_f1 = -1.0
    best_name = ""
    best_clf = None

    for name, clf in candidates.items():
        clf.fit(X_train_tfidf, y_train)

        y_pred = clf.predict(X_test_tfidf)

        acc = accuracy_score(y_test, y_pred)
        f1w = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        report = classification_report(
            y_test,
            y_pred,
            labels=labels.tolist(),
            output_dict=True,
            zero_division=0,
        )

        all_metrics[name] = {
            "accuracy": acc,
            "weighted_f1": f1w,
            "report": report,
        }

        if f1w > best_f1:
            best_f1 = f1w
            best_name = name
            best_clf = clf

    if best_clf is None:
        raise ValueError("Model training failed. No best model selected.")

    y_pred_best = best_clf.predict(X_test_tfidf)

    cm = confusion_matrix(
        y_test,
        y_pred_best,
        labels=labels.tolist(),
    )

    pipeline = Pipeline(
        [
            ("tfidf", vectorizer),
            ("clf", best_clf),
        ]
    )

    joblib.dump(best_clf, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(labels.tolist(), LABEL_PATH)

    return {
        "best_model_name": best_name,
        "pipeline": pipeline,
        "vectorizer": vectorizer,
        "clf": best_clf,
        "labels": labels.tolist(),
        "metrics": all_metrics[best_name],
        "all_metrics": all_metrics,
        "confusion": cm,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred_best": y_pred_best,
        "df_processed": df,
    }


def load_model() -> Tuple[Any, TfidfVectorizer, list] | None:
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        return None

    clf = joblib.load(MODEL_PATH)
    vec = joblib.load(VECTORIZER_PATH)
    labels = joblib.load(LABEL_PATH) if LABEL_PATH.exists() else []

    return clf, vec, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sentiment model")

    parser.add_argument("--csv", required=True)
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--label_col", default="sentiment")
    parser.add_argument("--lemmatize", action="store_true")

    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    results = train(
        df,
        args.text_col,
        args.label_col,
        lemmatize=args.lemmatize,
    )

    print(f"\n✅ Best model : {results['best_model_name']}")
    print(f"Accuracy     : {results['metrics']['accuracy']:.4f}")
    print(f"Weighted F1  : {results['metrics']['weighted_f1']:.4f}")
    print(f"Model saved  : {MODEL_PATH}")
