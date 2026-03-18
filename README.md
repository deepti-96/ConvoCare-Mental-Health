# ConvoCare - AI-Powered Mental Health Platform

> Built at **HackDuke 2021** (Code for Good - Health Track) · A semi-monitored mental health app for university students combining NLP, LSTM-based sentiment analysis, and privacy-first design.

---

##  The Problem

University students are facing a mental health crisis - and the system is failing them.

-  **Psychological distress** among students is at an all-time high
-  **Stigma** prevents students from seeking help from counselors
-  **Supply-demand mismatch** - students won't reach out, and institutions lack tools to proactively identify those in need
-  **Post-COVID isolation** has compounded the problem at scale

Existing solutions are reactive. ConvoCare is **proactive**.

---

##  The Solution

ConvoCare is a **privacy-first mental health companion** for university students. It gives students a safe space to express themselves, while providing passive, consent-based mental health insights - bridging the gap between students and support systems without forcing uncomfortable conversations.

### Core Features

- **Private Diary with On-Device Analysis** - Students maintain a personal diary of thoughts and emotions. Text is analyzed client-side using NLP to generate a mental health status report - raw diary content is never exposed.
 
- **LSTM Sentiment Classification** - A word-embedding LSTM model classifies entries as indicative of **anxiety**, **depression**, or neither - trained on Reddit mental health data with ~70% accuracy.
 
- ️ **Toxicity Detection** - Diary entries are analyzed for toxic language patterns using a pre-trained deep learning model, flagging high-risk content before it escalates.
 
- **Polarity & Subjectivity Scoring** - TextBlob-powered NLP pipeline computes emotional polarity and subjectivity scores to track mood trends over time.
 
- **Counseling Application** - Students can request counseling through the app - lowering the barrier of direct outreach with anonymized intake.

---

##  System Architecture

```
Student Input (Diary Entry)
         ↓
  Text Preprocessing
  (URL removal, stopwords, stemming, tokenization)
         ↓
     ┌───────────────────────────┐
     │  NLP Analysis Pipeline    │
     │  ├── Polarity Score       │  ← TextBlob
     │  ├── Subjectivity Score   │  ← TextBlob
     │  └── Toxicity Detection   │  ← LSTM Model (hDModel.h5)
     └───────────────────────────┘
         ↓
  Flask REST API (/endpoint)
         ↓
  Mental Health Status Report → Student Dashboard
```

---

##  ML Model - LSTM Sentiment Classifier

| Parameter | Value |
|---|---|
| Architecture | Bidirectional LSTM + GlobalMaxPool1D |
| Embedding Dimension | 128 |
| Max Features | 20,000 tokens |
| Max Sequence Length | 200 |
| Dropout | 0.2 |
| Output Classes | `anxiety`, `depression` |
| Training Set | 1,600 samples |
| Test Set | 352 samples |
| Accuracy | ~70% |
| Data Source | Reddit mental health posts (scraped via PRAW) |

### Model Pipeline

```python
Sequential([
    Input(shape=(200,)),
    Embedding(20000, 128, mask_zero=True),
    LSTM(64, return_sequences=True, dropout=0.2),
    GlobalMaxPool1D(),
    Dense(2, activation='sigmoid')  # anxiety, depression
])
```

### Why LSTM? - Approach Comparison

![ConvoCare](https://user-images.githubusercontent.com/72935128/139510770-cf3520b0-c01a-4074-bd70-4fdbdd01cd67.png)

---

### Data Preprocessing

1. Strip URLs, numbers, special characters
2. Lowercase normalization
3. NLTK stopword removal
4. Porter Stemmer tokenization
5. One-hot encode target labels (`anxiety`, `depression`)
6. Pad sequences to `maxlen=200`

---

##  Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask, Flask-CORS |
| ML / NLP | TensorFlow 2.4, Keras, LSTM, TextBlob, NLTK |
| Data Collection | Reddit Scraper (PRAW), BeautifulSoup |
| Server | Gunicorn, Gevent (WSGI) |
| Deployment | Heroku (Procfile) |
| Frontend | HTML, CSS (served via Flask templates) |

---

##  Project Structure

```
ConvoCare/
│
├── HackDuke.ipynb           # LSTM model training - data exploration, preprocessing, model
├── RedditScraper.ipynb      # Reddit data collection pipeline
│
└── server/
    ├── app.py               # Flask API - /endpoint + main route
    ├── sentiment_extraction.py  # TextBlob polarity & subjectivity
    ├── toxicity_analysis.py     # LSTM toxicity classifier
    ├── constants.py         # Config constants
    ├── requirements.txt     # Dependencies
    ├── Procfile             # Heroku deployment config
    └── static/              # Frontend assets
```

---

##  Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/deepti-96/ConvoCare-Mental-Health.git
cd ConvoCare-Mental-Health/server
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Server

```bash
python app.py
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### 4. API Usage

```bash
GET /endpoint?text=I+feel+overwhelmed+and+anxious
```

Returns:

```json
{
  "toxicity": ["anxiety"]
}
```

---

##  Future Roadmap

- **Improve model accuracy** - fine-tune with transformer-based models (BERT, RoBERTa)
- **Organization-wide deployment** - expand beyond universities to workplaces
- **Privacy-preserving activity tracking** - opt-in behavioral trend analysis
- **Social media integration** - mental health awareness outreach
- **Mood trend dashboard** - longitudinal visualization of student wellbeing

---

##  Social Impact

ConvoCare addresses **UN SDG 3 (Good Health and Well-Being)** by making mental health support accessible, stigma-free, and proactive for students who would otherwise go unserved.

---

##  Hackathon

**HackDuke 2021** - Code for Good · **Health Track**
Duke University · October 2021
