"""
fake.py — Safe Fake News Prediction
Works even if CNN model fails on Streamlit Cloud.
"""

import pickle
from pathlib import Path

import numpy as np

MAX_LEN = 300

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fake_news_cnn.h5"
TOKENIZER_PATH = BASE_DIR / "models" / "tokenizer.pkl"

_model = None
_tokenizer = None
_model_error = None


def load_assets():
    global _model, _tokenizer, _model_error

    try:
        if _model is None:
            from tensorflow.keras.models import load_model  # lazy import
            _model = load_model(MODEL_PATH, compile=False)

        if _tokenizer is None:
            with open(TOKENIZER_PATH, "rb") as f:
                _tokenizer = pickle.load(f)

        return _model, _tokenizer

    except Exception as e:
        _model_error = str(e)
        return None, None


def simple_fake_score(text):
    text = str(text).lower()

    fake_words = [
        "breaking", "shocking", "viral", "secret", "exposed",
        "miracle", "guaranteed", "100%", "click", "urgent",
        "fake", "rumor", "conspiracy"
    ]

    score = 0.0

    for word in fake_words:
        if word in text:
            score += 0.08

    if len(text.split()) < 5:
        score += 0.30

    if text.isupper():
        score += 0.20

    return min(score, 0.95)


def predict_fake_news(text):
    model, tokenizer = load_assets()

    # If CNN model loads successfully
    if model is not None and tokenizer is not None:
        from tensorflow.keras.preprocessing.sequence import pad_sequences  # lazy import

        seq = tokenizer.texts_to_sequences([str(text)])

        padded = pad_sequences(
            seq,
            maxlen=MAX_LEN,
            padding="post",
            truncating="post"
        )

        prob = float(model.predict(padded, verbose=0)[0][0])

    # Fallback if model fails
    else:
        prob = simple_fake_score(text)

    fake_prob = prob
    real_prob = 1.0 - prob

    if fake_prob >= 0.70:
        verdict = "FAKE 🚨"
    elif fake_prob >= 0.40:
        verdict = "SUSPICIOUS ⚠️"
    else:
        verdict = "REAL ✅"

    confidence = max(fake_prob, real_prob)

    return verdict, confidence, fake_prob, real_prob
