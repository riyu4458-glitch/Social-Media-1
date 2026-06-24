"""
predict.py — Prediction + explainability helpers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import clean_text


def predict_single(
    text: str,
    clf: Any,
    vectorizer: TfidfVectorizer,
    labels: List[str],
    lemmatize: bool = False,
) -> Dict[str, Any]:

    labels = list(labels)

    cleaned = clean_text(text, lemmatize=lemmatize)
    vec = vectorizer.transform([cleaned])

    try:
        proba = clf.predict_proba(vec)[0]
    except AttributeError:
        scores = clf.decision_function(vec)
        scores = np.asarray(scores)

        if scores.ndim == 1:
            scores = scores.reshape(1, -1)

        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        proba = (exp_scores / exp_scores.sum(axis=1, keepdims=True))[0]

    proba = np.asarray(proba).flatten()

    pred_idx = int(np.argmax(proba))
    pred_label = labels[pred_idx] if pred_idx < len(labels) else str(pred_idx)
    confidence = float(proba[pred_idx])

    probabilities = {}

    for i in range(min(len(labels), len(proba))):
        probabilities[labels[int(i)]] = float(proba[int(i)])

    top_words = _get_top_words(
        vec,
        vectorizer,
        clf,
        labels,
        pred_idx,
    )

    return {
        "label": pred_label,
        "confidence": confidence,
        "probabilities": probabilities,
        "top_words": top_words,
        "cleaned_text": cleaned,
    }


def predict_batch(
    df: pd.DataFrame,
    text_col: str,
    clf: Any,
    vectorizer: TfidfVectorizer,
    labels: List[str],
    lemmatize: bool = False,
) -> pd.DataFrame:

    labels = list(labels)

    cleaned = df[text_col].fillna("").apply(
        lambda t: clean_text(str(t), lemmatize=lemmatize)
    )

    vec = vectorizer.transform(cleaned)

    try:
        proba_matrix = clf.predict_proba(vec)
    except AttributeError:
        scores = clf.decision_function(vec)
        scores = np.asarray(scores)

        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)

        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        proba_matrix = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    proba_matrix = np.asarray(proba_matrix)

    pred_indices = np.argmax(proba_matrix, axis=1)

    pred_labels = [
        labels[int(i)] if int(i) < len(labels) else str(int(i))
        for i in pred_indices
    ]

    confidences = proba_matrix[
        np.arange(len(pred_indices)),
        pred_indices.astype(int)
    ]

    result = df.copy()
    result["predicted_sentiment"] = pred_labels
    result["confidence"] = confidences.round(4)

    for i, lbl in enumerate(labels):
        if i < proba_matrix.shape[1]:
            result[f"prob_{lbl}"] = proba_matrix[:, int(i)].round(4)

    return result


def _get_top_words(
    vec_row,
    vectorizer: TfidfVectorizer,
    clf: Any,
    labels: List[str],
    class_idx: int,
    top_n: int = 10,
) -> List[Tuple[str, float]]:

    feature_names = list(vectorizer.get_feature_names_out())
    tfidf_weights = np.asarray(vec_row.toarray()).flatten()

    nonzero_mask = tfidf_weights > 0

    if nonzero_mask.sum() == 0:
        return []

    try:
        clf_inner = clf

        if hasattr(clf_inner, "estimator"):
            clf_inner = clf_inner.estimator

        if hasattr(clf_inner, "base_estimator"):
            clf_inner = clf_inner.base_estimator

        if hasattr(clf_inner, "calibrated_classifiers_"):
            calibrated = clf_inner.calibrated_classifiers_
            if calibrated:
                clf_inner = calibrated[0].estimator

        coefs = clf_inner.coef_

        if coefs.shape[0] == 1:
            coef_row = coefs[0] if int(class_idx) == 1 else -coefs[0]
        else:
            coef_row = coefs[int(class_idx)]

        scores = coef_row * tfidf_weights

    except Exception:
        try:
            log_probs = clf.feature_log_prob_

            if log_probs.shape[0] > 1:
                other_idx = [
                    i for i in range(log_probs.shape[0])
                    if int(i) != int(class_idx)
                ]

                scores = (
                    log_probs[int(class_idx)]
                    - np.mean(log_probs[other_idx], axis=0)
                ) * tfidf_weights

            else:
                scores = log_probs[0] * tfidf_weights

        except Exception:
            scores = tfidf_weights

    scores = np.asarray(scores).flatten()
    scores[~nonzero_mask] = 0.0

    top_indices = np.argsort(scores)[::-1][:top_n]

    final_words = []

    for idx in top_indices:
        idx = int(idx)

        if idx < len(feature_names) and scores[idx] > 0:
            final_words.append(
                (
                    feature_names[idx],
                    float(scores[idx])
                )
            )

    return final_words
