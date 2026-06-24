"""
utils.py — Shared preprocessing and helper utilities.
"""

import re
import string
from typing import List, Optional

# ---------------------------------------------------------------------------
# Optional imports — gracefully degrade if not installed
# ---------------------------------------------------------------------------
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    # Download required NLTK data silently
    for pkg in ["stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

    STOP_WORDS = set(stopwords.words("english"))
    LEMMATIZER = WordNetLemmatizer()
    NLTK_AVAILABLE = True
except Exception:
    STOP_WORDS = set()
    LEMMATIZER = None
    NLTK_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    import os

    # Allow override via env var (useful on Windows)
    tess_cmd = os.getenv("TESSERACT_CMD", "")
    if tess_cmd:
        pytesseract.pytesseract.tesseract_cmd = tess_cmd
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str, lemmatize: bool = False) -> str:
    """
    Full preprocessing pipeline:
      1. Lowercase
      2. Remove URLs
      3. Remove special characters / digits
      4. Remove stopwords
      5. (Optional) Lemmatize
    """
    if not isinstance(text, str):
        text = str(text)

    # Lowercase
    text = text.lower()

    # Remove mentions, hashtags, URLs
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Remove special characters and digits
    text = re.sub(r"[^a-z\s]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenise (simple split)
    tokens = text.split()

    # Remove stopwords + context-ambiguous words that confuse sentiment models
    EXTRA_NEUTRAL = {
        "tried", "try", "use", "used", "using", "got", "get",
        "went", "go", "came", "come", "took", "take", "made", "make",
        "said", "say", "told", "tell", "found", "find", "seen", "see",
        "thing", "things", "time", "day", "way", "back", "just",
    }
    if NLTK_AVAILABLE and STOP_WORDS:
        tokens = [t for t in tokens if t not in STOP_WORDS and t not in EXTRA_NEUTRAL]
    elif EXTRA_NEUTRAL:
        tokens = [t for t in tokens if t not in EXTRA_NEUTRAL]

    # Lemmatize
    if lemmatize and NLTK_AVAILABLE and LEMMATIZER:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------

def extract_text_from_image(image) -> str:
    """
    Multi-pass OCR pipeline optimised for real screenshots including dark-mode.
    """
    if not OCR_AVAILABLE:
        return ""

    import cv2
    import numpy as np

    def _run_ocr(pil_img, config="--psm 6 --oem 3"):
        try:
            return pytesseract.image_to_string(pil_img, config=config).strip()
        except Exception:
            return ""

    def _is_dark(gray_img):
        return np.mean(gray_img) < 127

    results = []
    try:
        img = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        scale = 2
        up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Auto-invert dark backgrounds so text becomes dark-on-light
        if _is_dark(up):
            up = cv2.bitwise_not(up)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(up, h=10)

        # Pass 1: plain upscaled
        results.append(_run_ocr(Image.fromarray(up)))

        # Pass 2: denoised
        results.append(_run_ocr(Image.fromarray(denoised)))

        # Pass 3: OTSU
        _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(_run_ocr(Image.fromarray(otsu)))

        # Pass 4: adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
        )
        results.append(_run_ocr(Image.fromarray(adaptive)))

        # Pass 5: sharpen
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        results.append(_run_ocr(Image.fromarray(sharpened)))

        # Pass 6: sparse text mode
        results.append(_run_ocr(image, "--psm 11 --oem 3"))

    except Exception:
        results.append(_run_ocr(image))

    best = max(results, key=lambda t: len(t.split()))
    return best


# ---------------------------------------------------------------------------
# Label normalisation
# ---------------------------------------------------------------------------

POSITIVE_LABELS = {
    # Direct
    "positive", "positivity", "pos",
    # Joy / Happiness
    "joy", "happiness", "happy", "elation", "ecstasy", "euphoria",
    "overjoyed", "festivejoy", "joyfulreunion", "joy in baking", "playfujoy",
    # Love / Affection
    "love", "affection", "adoration", "romance", "tenderness", "heartwarming",
    # Excitement / Energy
    "excitement", "enthusiasm", "adrenaline", "thrill", "thrilling journey",
    "energy", "vibrancy", "zest", "spark",
    # Gratitude / Appreciation
    "gratitude", "grateful", "appreciation", "blessed",
    # Calm / Peace
    "calmness", "serenity", "tranquility", "harmony", "peace", "mindfulness",
    "rejuvenation", "solace",
    # Achievement
    "accomplishment", "triumph", "success", "pride", "proud", "empowerment",
    "confidence", "confident", "fulfillment", "satisfaction", "breakthrough",
    # Positive emotions
    "admiration", "awe", "wonder", "wonderment", "amazement", "captivation",
    "enchantment", "mesmerizing", "hypnotic", "dazzle", "radiance",
    "inspiration", "inspired", "motivation", "optimism", "hope", "hopeful",
    "determination", "resilience",
    # Social / Creative
    "friendship", "connection", "compassion", "compassionate", "kindness",
    "kind", "sympathy", "empathetic", "celebration",
    "creativity", "creative inspiration", "artisburst", "imagination",
    "amusement", "playful", "playfujoy", "mischievous", "whimsy",
    "exploration", "adventure", "freedom", "free-spirited",
    # Misc positive
    "charm", "elegance", "iconic", "grandeur", "colorful", "melodic",
    "acceptance", "contentment", "coziness", "enjoyment", "engagement",
    "immersion", "intrigue", "curiosity", "anticipation",
    "relief", "touched", "heartwarming", "nostalgia", "bittersweet",
    "dreamchaser", "innerjou rney", "renewed effort",
}

NEGATIVE_LABELS = {
    # Direct
    "negative", "neg", "bad",
    # Sadness
    "sad", "sadness", "sorrow", "grief", "melancholy", "heartbreak",
    "heartache", "lostlove", "loss", "desolation", "despair",
    "loneliness", "isolation", "numbness", "suffering",
    # Anger
    "anger", "hate", "resentment", "bitterness", "bitter", "betrayal",
    "envy", "envious", "jealousy", "jealous",
    # Fear / Anxiety
    "fear", "fearful", "anxiety", "apprehensive", "intimidation",
    "suspense", "darkness",
    # Frustration / Failure
    "frustration", "frustrated", "disappointment", "disappointed",
    "disgust", "shame", "embarrassed", "regret", "helplessness",
    "devastated", "desperation", "overwhelmed", "exhaustion", "pressure",
    "obstacle", "miscalculation",
    # Dismissive
    "dismissive", "indifference", "boredom", "pensive", "ambivalence",
    "confusion", "isolation", "ruins",
}


def normalise_label(raw: str) -> str:
    """Map raw emotion labels → Positive / Neutral / Negative."""
    cleaned = str(raw).lower().strip()
    if cleaned in POSITIVE_LABELS:
        return "Positive"
    if cleaned in NEGATIVE_LABELS:
        return "Negative"
    return "Neutral"


# ---------------------------------------------------------------------------
# Availability flags (consumed by app.py)
# ---------------------------------------------------------------------------

def check_ocr() -> bool:
    return OCR_AVAILABLE


def check_nltk() -> bool:
    return NLTK_AVAILABLE
