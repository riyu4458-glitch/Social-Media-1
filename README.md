# 🧠 SentimentIQ — Social Media Sentiment Analytics Platform

AI-powered dashboard built using Streamlit.

## Features

✔ Sentiment Model Training  
✔ Single Text Sentiment Prediction  
✔ Fake News Detection  
✔ OCR + Sentiment Detection  
✔ Analytics Dashboard  

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- TensorFlow
- Plotly
- OpenCV
- Tesseract OCR

## Installation

```bash
git clone https://github.com/yourusername/SentimentIQ.git

cd SentimentIQ

pip install -r requirements.txt

streamlit run app.py
```
## 📂 Project Structure

```text
SentimentIQ/
│
├── app.py
├── train.py
├── predict.py
├── fake.py
├── utils.py
├── requirements.txt
├── README.md
├── .gitignore
├── LICENSE
│
├── models/
│   ├── sentiment_model.pkl
│   ├── vectorizer.pkl
│   └── labels.pkl
│
├── data/
│   ├── sample_dataset.csv
│   └── fake_news_sample.csv
│
├── assets/
│   ├── dashboard.png
│   ├── banner.png
│   └── logo.png
│
├── notebooks/
│   ├── model_training.ipynb
│   └── experiments.ipynb
│
└── docs/
    ├── architecture.png
    └── workflow.png
```

## Dataset

CSV format:

| text | sentiment |
|------|-----------|
| Great app | Positive |

## Screenshots

Add images inside `/assets`

