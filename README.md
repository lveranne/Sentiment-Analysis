# 📉 NLP Social Insights: Trend & Sentiment Detection with LSTM and Flask

This project implements a **Twitter sentiment analysis** pipeline using a **Recurrent Neural Network (LSTM)** trained on tweets about various technologies (games, apps, software), and a **Flask web application** to serve real-time predictions.

Given a tweet or short text, the model predicts one of four sentiment classes:

- **Positive**
- **Negative**
- **Neutral**
- **Irrelevant**

The app includes preprocessing (cleaning, normalization, stopword removal), language detection and automatic translation to English, and returns a sentiment label along with an emoji.

---

### Features

- **End-to-end sentiment analysis**
  - Text cleaning (HTML tags, URLs, punctuation, numbers, extra whitespace, emojis)
  - Tokenization and stopword removal (NLTK)
  - Sequence vectorization for deep learning models
- **LSTM-based model**
  - Trained using Keras/TensorFlow on `twitter.csv`
- **Language-aware**
  - Detects input language with `langdetect`
  - Translates non-English text to English using `googletrans` before prediction
- **Web interface (Flask)**
  - Simple form to input a tweet/comment
  - Displays predicted sentiment and an emoji:
    - Positive: 😄
    - Negative: 😞
    - Neutral: 😐
    - Irrelevant: 🤷‍♂️
- **Notebook for analysis & training**
  - `Analyse_Des_Sentiments.ipynb` contains:
    - Data exploration and description
    - Preprocessing
    - Model definition and training
    - Evaluation (accuracy, confusion matrix, etc.)

---

### Project Structure

- **`app.py`** – Flask application exposing the web interface and prediction endpoint.
- **`Analyse_Des_Sentiments.ipynb`** – Jupyter notebook for data analysis, preprocessing, model training, and evaluation.
- **`model_lstm.keras`** – Saved LSTM model (Keras format).
- **`twitter.csv`** – Dataset of tweets and associated sentiments.

You may also add a `templates/` folder with at least:

- `templates/index.html` – Main page with a form to enter a tweet/comment.
- `templates/result.html` – Result page displaying the prediction and emoji.

---

### Data Description

The dataset `twitter.csv` contains sentiment-labelled tweets about various technologies.

**Columns:**

1. **`Technology`** – Names of video games, applications, or software.
2. **`Sentiment`** – One of:
   - `Positive`
   - `Negative`
   - `Neutral`
   - `Irrelevant`
3. **`Tweet`** – The text content of the tweet or message.

The notebook `Analyse_Des_Sentiments.ipynb` shows how this data is loaded, explored, cleaned, and used to train the LSTM model.

---

### Model Overview

The core model is an **LSTM-based neural network** built with TensorFlow / Keras:

- Text is vectorized into sequences of integers using a `Tokenizer` (with a vocabulary size of about 10,000 tokens).
- Sequences are padded/truncated to a fixed length (e.g. 100 tokens).
- The model uses recurrent layers (LSTM / Bidirectional LSTM, etc. as defined in the notebook) and Dense layers for classification into 4 sentiment classes.
- The trained model is saved as `model_lstm.keras` and loaded in `app.py` for inference.

Training, validation, and evaluation details (accuracy, confusion matrix, word clouds, etc.) are documented in the notebook.

---

### Preprocessing Pipeline

The following preprocessing is applied in `app.py` before prediction:

1. **Language detection & translation**
   - `langdetect.detect(text)` to infer language.
   - If not English, `googletrans.Translator` translates to English.

2. **Text normalization and cleaning**
   - Lowercasing
   - Removing HTML tags
   - Removing URLs
   - Collapsing extra whitespace
   - Removing numbers
   - Removing punctuation
   - Removing emojis and a range of special Unicode symbols

3. **Tokenization and stopword removal**
   - `nltk.word_tokenize` for word tokenization.
   - NLTK English stopwords are removed.

4. **Vectorization**
   - Text is transformed to integer sequences via a Keras `Tokenizer`.
   - Sequences are padded to a fixed length (e.g. 100) using `pad_sequences`.

---

### Tech Stack

- **Language**: Python
- **Web framework**: Flask
- **Deep Learning**: TensorFlow / Keras
- **NLP & preprocessing**:
  - NLTK
  - langdetect
  - googletrans
- **Data & visualization** (mainly in the notebook):
  - pandas
  - NumPy
  - scikit-learn
  - matplotlib, seaborn
  - wordcloud

---

### Installation

#### 1. Clone the repository

git clone https://github.com/<your-username>/Twitter-Sentiment-Analysis.git
cd Twitter-Sentiment-Analysis

