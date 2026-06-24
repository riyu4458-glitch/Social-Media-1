"""
app.py — Sentiment Analysis Intelligence Dashboard
5-page Streamlit application for ML portfolio showcase.
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytesseract
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import streamlit as st

from predict import predict_batch, predict_single
from train import load_model, train
from utils import check_nltk, check_ocr, clean_text, extract_text_from_image

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SentimentIQ",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
# =====================================================
# SIDEBAR STYLE
# =====================================================
st.markdown("""
<style>
section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#042a52,#0a3d91);

    width:420px !important;
    min-width:420px !important;
    max-width:420px !important;

    border-right:1px solid rgba(255,255,255,.08);
}

/* Inner sidebar container */
section[data-testid="stSidebar"] > div{
    width:420px !important;
    min-width:420px !important;
}

/* SentimentIQ title */
.sidebar-title{

    font-size:60px !important;

    font-weight:900 !important;

    white-space:nowrap !important;

    color:white !important;

    margin-bottom:25px !important;
}

/* Navigation text */
div[role="radiogroup"] label{

    font-size:24px !important;

    font-weight:700 !important;

    padding-top:12px !important;

    padding-bottom:12px !important;

    color:white !important;
}

/* Sidebar paragraphs */
section[data-testid="stSidebar"] p{
    font-size:20px !important;
}



/* ==========================================
   GLOBAL
========================================== */

.stApp{
background:
radial-gradient(circle at top left,
rgba(124,58,237,.35),
transparent 35%),

radial-gradient(circle at top right,
rgba(6,182,212,.25),
transparent 35%),

linear-gradient(
135deg,
#021024,
#052a5a,
#0a3d91
);

color:white;
}

.block-container{
max-width:1450px;
padding-top:2rem;
padding-bottom:2rem;
}

/* Hide Streamlit Branding */

#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header{visibility:hidden;}


/* ==========================================
   HEADINGS
========================================== */

h1{
font-size:3rem !important;
font-weight:900 !important;
color:white !important;
}

h2{
color:#06B6D4 !important;
}

h3{
color:#60A5FA !important;
}

/* ==========================================
   METRICS
========================================== */

div[data-testid="metric-container"]{

background:#0F172A;

padding:20px;

border-radius:20px;

border:1px solid rgba(255,255,255,.08);

box-shadow:
0 10px 25px rgba(0,0,0,.35);

transition:.3s;
}

div[data-testid="metric-container"]:hover{

transform:translateY(-5px);

box-shadow:
0 20px 40px rgba(124,58,237,.25);
}

/* ==========================================
   BUTTONS
========================================== */

.stButton button{

width:100%;

height:52px;

border:none;

border-radius:14px;

font-weight:700;

font-size:15px;

color:white;

background:
linear-gradient(
135deg,
#2563EB,
#06B6D4
);

transition:.3s;
}

.stButton button:hover{

transform:translateY(-3px);

box-shadow:
0 15px 35px rgba(124,58,237,.4);
}

/* ==========================================
   INPUTS
========================================== */

.stTextArea textarea,
.stTextInput input{

background:#0F172A !important;

color:white !important;

border:1px solid #2563EB !important;

border-radius:15px !important;
}

/* ==========================================
   FILE UPLOADER
========================================== */

[data-testid="stFileUploader"]{

background:#0F172A;

border:2px dashed #3B82F6;

border-radius:20px;

padding:10px;
}

/* ==========================================
   ALERTS
========================================== */

[data-testid="stAlert"]{

background:
rgba(124,58,237,.12) !important;

border-left:5px solid #3B82F6 !important;

border-radius:15px !important;

color:white !important;
}

/* ==========================================
   DATAFRAMES
========================================== */

[data-testid="stDataFrame"]{

border-radius:15px;

overflow:hidden;

border:1px solid rgba(255,255,255,.08);
}

/* ==========================================
   PROGRESS BAR
========================================== */

.stProgress > div > div > div > div{

background:
linear-gradient(
90deg,
#06B6D4,
#3B82F6
);
}

/* ==========================================
   ANIMATION
========================================== */

@keyframes fadeUp{

from{
opacity:0;
transform:translateY(15px);
}

to{
opacity:1;
transform:translateY(0);
}
}

section.main > div{
animation:fadeUp .5s ease;
}

</style>
""", unsafe_allow_html=True)


 

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
for key, default in {
    "clf": None,
    "vec": None,
    "labels": [],
    "best_model_name": "",
    "train_results": None,
    "history": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

OCR_OK = check_ocr()
NLTK_OK = check_nltk()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.sidebar.markdown("""
    <h1 style='
    text-align:center;
    color:white;
   font-size:32px;
   font-weight:800;
   white-space:nowrap;
   margin-bottom:5;
   '>
   SentimentIQ
       🧠
   </h1>
   """, unsafe_allow_html=True)
    st.divider()
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio(
        "",
        [
            "🏋️ Train Model",
            "💬 Single Text",
            "🤨 Fake News Detection",
            "🖼️ Image OCR",
            "📊 Analytics",
        ],
    )
    # ==========================================================
# GLOBAL PAGE HEADER
# ==========================================================

def show_header(title, subtitle):

    st.markdown(f"""
    <div style="
    background:linear-gradient(135deg,#021024,#0A3D91);
    padding:30px;
    border-radius:20px;
    margin-bottom:30px;
    box-shadow:0 10px 25px rgba(0,140,255,.25);
    ">

    <h1 style="
    color:white;
    margin:0;
    font-size:46px;
    ">
    {title}
    </h1>

    <p style="
    color:#D6E8FF;
    margin-top:15px;
    font-size:18px;
    ">
    {subtitle}
    </p>

    </div>
    """, unsafe_allow_html=True)


if page == "🏋️ Train Model":

    show_header(
        "🚀 Social Media Analytics Platform",
        "Analyze Sentiment • Detect Fake News • OCR Images "
    )

elif page == "💬 Single Text":

    show_header(
        "💬 Sentiment Analysis Dashboard",
        "Deep Learning Sentiment Classification"
    )


elif page == "🖼️ Image OCR":

    show_header(
        "🖼 OCR Intelligence Dashboard",
        "Extract & Analyze Text from Images"
    )

elif page == "📊 Analytics":

    show_header(
        "📊 Analytics Dashboard",
        "Insights • Metrics • Visualization"
    )
    st.divider()
    # Status indicators
    model_ready = st.session_state.clf is not None
    #st.markdown(f"**Model loaded:** {'✅' if model_ready else '⭕ train first'}")
    #st.markdown(f"**OCR (Tesseract):** {'✅' if OCR_OK else '❌ not installed'}")
    #st.markdown(f"**Fake News Detector:** {'✅' if NLTK_OK else '⚠️ not found'}")

    if model_ready:
        st.caption(f"Active: **{st.session_state.best_model_name}**")

    # Try loading saved model on startup
    if not model_ready:
        saved = load_model()
        if saved:
            st.session_state.clf, st.session_state.vec, st.session_state.labels = saved
            st.session_state.best_model_name = "Saved model"
            st.success("Saved model auto-loaded ✅")
st.sidebar.markdown("---")
# ---------------------------------------------------------------------------
# Shared colour helpers
# ---------------------------------------------------------------------------

SENT_COLORS = {"Positive": "#00c9a7", "Neutral": "#f0a500", "Negative": "#f25c54"}


def _badge(label: str, conf: float) -> str:
    color = SENT_COLORS.get(label, "#7c6af7")
    return (
        f'<span style="background:{color};color:#000;padding:4px 14px;'
        f'border-radius:20px;font-weight:700;font-size:1.1rem;">'
        f"{label}</span> &nbsp; <span style='font-size:.9rem;color:#ccc;'>"
        f"Confidence: {conf:.1%}</span>"
    )


def _prob_bars(probs: dict) -> None:
    """Render horizontal probability bars using Plotly."""
    labels_list = list(probs.keys())
    values = list(probs.values())
    colors = [SENT_COLORS.get(l, "#7c6af7") for l in labels_list]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels_list,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e4e6f0",
        margin=dict(l=10, r=10, t=10, b=10),
        height=180,
    )
    st.plotly_chart(fig, use_container_width=True)


def _top_words_chart(top_words: list) -> None:
    if not top_words:
        st.info("No influential words found (text may be too short).")
        return
    words, scores = zip(*top_words)
    fig = go.Figure(
        go.Bar(
            x=scores,
            y=words,
            orientation="h",
            marker_color="#7c6af7",
            marker_opacity=0.85,
        )
    )
    fig.update_layout(
        title="Top words influencing this prediction",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e4e6f0",
        margin=dict(l=10, r=30, t=40, b=10),
        height=max(200, 35 * len(words)),
    )
    st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE 1 — Train Model
# ===========================================================================
if page == "🏋️ Train Model":
    #st.header("🏋️ Train a Sentiment Model")

    # ── Upload & configure ──────────────────────────────────────────────────
    uploaded = st.file_uploader("Upload labeled CSV dataset", type=["csv"])

    if uploaded:
        try:
            import io
            raw_bytes = uploaded.read()
            first_val = pd.read_csv(io.BytesIO(raw_bytes), nrows=1, header=None).iloc[0, 0]
            has_header = not str(first_val).strip().lstrip('-').isdigit()
            df_raw = pd.read_csv(io.BytesIO(raw_bytes), header=0 if has_header else None)
            if not has_header:
                df_raw.columns = [f"col_{i}" for i in range(len(df_raw.columns))]
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")
            st.stop()

        st.success(f"Loaded {len(df_raw):,} rows × {len(df_raw.columns)} columns")
        st.dataframe(df_raw.head(5), use_container_width=True)

        cols = df_raw.columns.tolist()

        # Auto-detect common column names from popular datasets
        def _guess(options, candidates):
            for c in candidates:
                if c in options:
                    return options.index(c)
            return 0

        text_default = _guess(cols, ["text", "tweet", "Text", "Tweet", "sentence", "review"])
        label_default = _guess(cols, ["sentiment", "Sentiment", "label", "Label", "class", "emotion"])

        c1, c2 = st.columns(2)
        text_col = c1.selectbox("Text column", cols, index=text_default)
        label_col = c2.selectbox("Label / Sentiment column", cols, index=label_default)
        lemmatize = st.checkbox("Enable lemmatization (slower, sometimes better)")

        if st.button("🚀 Train & Compare Models", type="primary"):
            with st.spinner("Training three models — this may take a moment…"):
                try:
                    results = train(
                        df_raw,
                        text_col=text_col,
                        label_col=label_col,
                        lemmatize=lemmatize,
                    )
                except Exception as e:
                    st.error("Training failed")
                    st.exception(e)
                    st.stop()

            # Store in session
            st.session_state.clf = results["clf"]
            st.session_state.vec = results["vectorizer"]
            st.session_state.labels = results["labels"]
            st.session_state.best_model_name = results["best_model_name"]
            st.session_state.train_results = results

            st.success(f"✅ Best model: **{results['best_model_name']}**")

            # ── Comparison table ──
            st.subheader("Model comparison")
            rows = []
            for name, m in results["all_metrics"].items():
                r = m["report"]
                rows.append(
                    {
                        "Model": name,
                        "Accuracy": f"{m['accuracy']:.4f}",
                        "Weighted F1": f"{m['weighted_f1']:.4f}",
                        "Precision (w)": f"{r['weighted avg']['precision']:.4f}",
                        "Recall (w)": f"{r['weighted avg']['recall']:.4f}",
                    }
                )
            cmp_df = pd.DataFrame(rows)
            st.dataframe(cmp_df, use_container_width=True, hide_index=True)

            # ── Bar chart of F1 ──
            fig = px.bar(
                cmp_df,
                x="Model",
                y="Weighted F1",
                color="Weighted F1",
                color_continuous_scale=["#f25c54", "#f0a500", "#00c9a7"],
                title="Weighted F1 Score — Model Comparison",
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e4e6f0",
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info(
            "Upload a CSV with at least two columns: one with text, one with sentiment labels.\n\n"
            "**Expected label values:** Positive, Negative, Neutral (or raw emotions — they'll be mapped automatically)."
        )
        st.markdown(
            """
**Example dataset format:**

| text | sentiment |
|------|-----------|
| I love this product! | Positive |
| Terrible experience. | Negative |
| It was okay I guess. | Neutral |
"""
        )


# ===========================================================================
# PAGE 2 — Single Text Prediction
# ===========================================================================
elif page == "💬 Single Text":
    #st.header("💬 Single Text Prediction")

    if not st.session_state.clf:
        st.warning("Train a model first (or ensure a saved model exists in `/models`).")
        st.stop()

    user_text = st.text_area(
        "Enter text to analyse",
        placeholder="Type or paste a social media post, review, or any text…",
        height=120,
    )

    if st.button("🔍 Predict Sentiment", type="primary") and user_text.strip():
        result = predict_single(
            user_text,
            st.session_state.clf,
            st.session_state.vec,
            st.session_state.labels,
        )

        # Store in history
        st.session_state.history.append(
            {
                "text": user_text[:80] + ("…" if len(user_text) > 80 else ""),
                "label": result["label"],
                "confidence": result["confidence"],
            }
        )

        st.divider()
        c1, c2 = st.columns([1, 1])

        with c1:
            st.subheader("Prediction")
            st.markdown(_badge(result["label"], result["confidence"]), unsafe_allow_html=True)
            st.markdown("**Sentiment probabilities**")
            _prob_bars(result["probabilities"])

            st.markdown("**Cleaned text seen by the model**")
            st.code(result["cleaned_text"] or "(empty after cleaning)", language=None)

        with c2:
            st.subheader("Word Influence")
            _top_words_chart(result["top_words"])
            st.caption(
                "Each bar shows how strongly a word pushed the model toward the predicted class. "
                "Score = model coefficient × TF-IDF weight."
            )


# ===========================================================================
# PAGE 3 — Fake News Detection
# ===========================================================================

elif page == "🤨 Fake News Detection":

    st.markdown("""
    <div style="
    background:linear-gradient(135deg,#021024,#0A3D91);
    padding:30px;
    border-radius:20px;
    margin-bottom:25px;
    box-shadow:0 10px 25px rgba(0,140,255,.25);
    ">

    <h1 style="
    color:white;
    margin:0;
    font-size:44px;
    ">
    📰 Fake News Intelligence Dashboard
    </h1>

    <h4 style="
    color:#D6E8FF;
    margin-top:15px;
    margin-bottom:10px;
    ">
    CNN-based Fake News Detection
    </h4>

    <p style="
    color:#A5D8FF;
    font-size:18px;
    margin:0;
    ">
    Detect fake • suspicious 
    </p>

    </div>
    """, unsafe_allow_html=True)

    try:
        from fake import predict_fake_news

    except Exception as e:
        st.error(f"Error importing app_3.py: {e}")
        st.stop()

    if "history" not in st.session_state:
        st.session_state.history = []

    news_text = st.text_area(
        "Enter News Text",
        height=220,
        placeholder="Paste headline, article, tweet, or news content...",
        key="fake_news_input"
    )

    analyze = st.button(
        "🔍 Analyze News",
        use_container_width=True,
        key="fake_news_btn"
    )

    if analyze:

        if not news_text.strip():
            st.warning("Enter some text first.")

        else:

            verdict, confidence, fake_prob, real_prob = predict_fake_news(news_text)

            st.session_state.history.append({
                "text": news_text[:80],
                "label": verdict,
                "confidence": confidence,
            })

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Verdict", verdict)
            c2.metric("Confidence", f"{confidence*100:.2f}%")
            c3.metric("Fake %", f"{fake_prob*100:.2f}%")
            c4.metric("Real %", f"{real_prob*100:.2f}%")

            st.write("Fake Probability")
            st.progress(fake_prob)

            st.write("Real Probability")
            st.progress(real_prob)

            if "FAKE" in verdict:
                st.error(f"🚨 {verdict}")

            elif "SUSPICIOUS" in verdict:
                st.warning(f"⚠️ {verdict}")

            else:
                st.success(f"✅ {verdict}")
#===========================================================================
#PAGE 4 — Image OCR
# ===========================================================================
elif page == "🖼️ Image OCR":
    #st.header("🖼️ Image Sentiment Analysis")

    if not OCR_OK:
        st.error(
            "Tesseract OCR is not installed or pytesseract could not find it.\n\n"
            "**Install instructions:**\n"
            "- Linux: `sudo apt-get install tesseract-ocr`\n"
            "- macOS: `brew install tesseract`\n"
            "- Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki\n\n"
            "After installation, restart Streamlit."
        )
        st.stop()

    if not st.session_state.clf:
        st.warning("Train or load a model before predicting.")
        st.stop()

    uploaded_img = st.file_uploader(
        "Upload an image (screenshot, meme, etc.)", type=["png", "jpg", "jpeg", "webp"]
    )

    if uploaded_img:
        from PIL import Image

        image = Image.open(uploaded_img)
        c1, c2 = st.columns([1, 1])

        with c1:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with c2:
            with st.spinner("Running multi-pass OCR…"):
                import cv2, numpy as np
                img_arr = np.array(image.convert("RGB"))
                gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
                if np.mean(gray) < 127:
                    gray = cv2.bitwise_not(gray)
                up = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                denoised = cv2.fastNlMeansDenoising(up, h=10)
                _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                cfg = "--psm 6 --oem 3"
                results = []
                for arr in [up, denoised, otsu]:
                    try:
                        import pytesseract
                        t = pytesseract.image_to_string(Image.fromarray(arr), config=cfg).strip()
                        results.append(t)
                    except Exception:
                        pass
                extracted = max(results, key=lambda t: len(t.split())) if results else ""

            if extracted and len(extracted.split()) >= 2:
                st.subheader("Extracted Text")
                st.text_area("OCR output", extracted, height=180, disabled=True)
                st.caption(f"Words extracted: {len(extracted.split())}")
                result = predict_single(
                    extracted,
                    st.session_state.clf,
                    st.session_state.vec,
                    st.session_state.labels,
                )
                st.divider()
                st.subheader("Prediction")
                st.markdown(
                    _badge(result["label"], result["confidence"]),
                    unsafe_allow_html=True,
                )
                _prob_bars(result["probabilities"])
                _top_words_chart(result["top_words"])
            else:
                st.warning(
                    "Could not extract enough text.\n\n"
                    "**Tips:**\n"
                    "- Light background, dark text works best\n"
                    "- Dark-mode screenshots are auto-inverted — try uploading as-is\n"
                    "- Use PNG over JPEG\n"
                    "- Crop to the text area only"
                )


# ===========================================================================
# PAGE 5 — Analytics Dashboard
# ===========================================================================
elif page == "📊 Analytics":
    #st.header("📊 Analytics Dashboard")

    results = st.session_state.train_results

    if not results:
        st.warning("Train a model first to see analytics.")
        st.stop()

    df_proc = results["df_processed"]
    labels = results["labels"]

    # ── Row 1: key metrics ──────────────────────────────────────────────────
    best_metrics = results["metrics"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best Model", results["best_model_name"])
    m2.metric("Accuracy", f"{best_metrics['accuracy']:.2%}")
    m3.metric("Weighted F1", f"{best_metrics['weighted_f1']:.4f}")
    m4.metric("Training Samples", f"{len(df_proc):,}")

    st.divider()

    # ── Row 2: class distribution + confusion matrix ────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Class Distribution")
        counts = df_proc["label"].value_counts()
        fig = px.bar(
            x=counts.index,
            y=counts.values,
            color=counts.index,
          
            color_discrete_map=SENT_COLORS,
            labels={"x": "Sentiment", "y": "Count"},
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e4e6f0",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Confusion Matrix")
        cm = results["confusion"]
        fig = px.imshow(
            cm,
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale=[[0, "#1a1d27"], [1, "#00c9a7"]],
            labels=dict(x="Predicted", y="Actual"),
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e4e6f0",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: model comparison ─────────────────────────────────────────────
    st.subheader("Model Comparison")
    rows = []
    for name, m in results["all_metrics"].items():
        r = m["report"]
        rows.append(
            {
                "Model": name,
                "Accuracy": m["accuracy"],
                "Weighted F1": m["weighted_f1"],
                "Precision": r["weighted avg"]["precision"],
                "Recall": r["weighted avg"]["recall"],
            }
        )
    cmp_df = pd.DataFrame(rows)

    metrics_long = cmp_df.melt(
        id_vars="Model", var_name="Metric", value_name="Score"
    )
    fig = px.bar(
        metrics_long,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        color_discrete_sequence=["#0EA5E9","#2563EB","#1D4ED8"],
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e4e6f0",
        yaxis=dict(range=[0, 1]),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Row 4: top TF-IDF words overall ────────────────────────────────────
    st.subheader("Top TF-IDF Words by Sentiment Class")

    vec = st.session_state.vec
    clf = st.session_state.clf
    feature_names = vec.get_feature_names_out()

    tabs = st.tabs(labels)
    for tab, label in zip(tabs, labels):
        with tab:
            try:
                class_idx = labels.index(label)
                # Unwrap calibrated or pipeline wrappers
                clf_inner = clf
                for attr in ("estimator", "base_estimator", "calibrated_classifiers_"):
                    if hasattr(clf_inner, attr):
                        inner = getattr(clf_inner, attr)
                        clf_inner = inner[0].estimator if isinstance(inner, list) else inner
                        break
                if hasattr(clf_inner, "coef_"):
                    coef = clf_inner.coef_
                    coef_row = coef[0] if coef.shape[0] == 1 else coef[class_idx]
                elif hasattr(clf_inner, "feature_log_prob_"):
                    lp = clf_inner.feature_log_prob_
                    others = [i for i in range(lp.shape[0]) if i != class_idx]
                    coef_row = lp[class_idx] - lp[others].mean(axis=0)
                else:
                    st.info("Coefficient extraction not available for this model.")
                    continue
                top20_idx = np.argsort(coef_row)[::-1][:20]
                top20_words = feature_names[top20_idx]
                top20_scores = coef_row[top20_idx]
            except Exception as e:
                st.info(f"Could not extract coefficients: {e}")
                continue

            fig = go.Figure(
                go.Bar(
                    x=top20_scores[::-1],
                    y=top20_words[::-1],
                    orientation="h",
                    marker_color=SENT_COLORS.get(label,"#2563EB"),
                    marker_opacity=0.85,
                )
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e4e6f0",
                margin=dict(l=10, r=10, t=10, b=10),
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)
   
    # ── Prediction history ──────────────────────────────────────────────────
    if st.session_state.history:
        st.subheader("Session Prediction History")
        hist_df = pd.DataFrame(st.session_state.history)
        hist_df["confidence"] = hist_df["confidence"].map("{:.1%}".format)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        csv_bytes = pd.DataFrame(st.session_state.history).to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download History CSV",
            data=csv_bytes,
            file_name="session_history.csv",
            mime="text/csv",
        )
