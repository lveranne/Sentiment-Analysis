from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)
label_encoder = LabelEncoder()
model = load_model('model_lstm.keras')
tokenizer = Tokenizer(num_words=10000)
#tokenizer.fit_on_texts(comments)

def normalize_text(text):
    return text.lower()
def remove_white_spaces(text):
    return re.sub(r'\s+', ' ', text, flags=re.I)
def remove_html_tags(text):
    return re.sub(r'<.*?>', '', text)
def remove_urls(text):
    return re.sub(r'http\S+|www\S+', '', text)
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))
def remove_numbers(text):
    return re.sub(r'\d+', '', text)
def remove_emojis(text):
    if isinstance(text, str):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002500-\U00002BEF"  # chinese char
                                   u"\U00002702-\U000027B0"
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   u"\U0001f926-\U0001f937"
                                   u"\U00010000-\U0010ffff"
                                   u"\u2640-\u2642"
                                   u"\u2600-\u2B55"
                                   u"\u200d"
                                   u"\u23cf"
                                   u"\u23e9"
                                   u"\u231a"
                                   u"\ufe0f"  # dingbats
                                   u"\u3030"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    else:
        return text
def tokenize_text(text):
    return word_tokenize(text)
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

from langdetect import detect
from googletrans import Translator
def detect_trans(text):
    lang = detect(text)
    if lang != 'en':
        translator = Translator()
        text = translator.translate(text, dest='en').text
        return text
    else:
        return text
def preprocess(text):
    text = detect_trans(text)
    text = normalize_text(text)
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_white_spaces(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = remove_emojis(text)
    return text
def vectorize_data(X, maxlen):
    maxlen= 100
    X = tokenizer.texts_to_sequences(X)
    X = pad_sequences(X, maxlen = maxlen)
    return X

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sentiment_emojis = {
    'Positive': '😄',
    'Negative': '😞',
    'Neutral': '😐',
    'Irrelevant': '🤷‍♂️'
}
    comment = request.form['comment']
    
    comment = preprocess(comment)
    
    comment = tokenize_text(comment)
    
    comment = remove_stopwords(comment)
    
    comment =[comment]
    
    comment_vect = vectorize_data(comment)
    
    probs = model.predict(comment_vect)
    
    prediction = np.argmax(probs, axis=1)
    
    sentiment = label_encoder.inverse_transform(prediction)
    
    emoji = sentiment_emojis[sentiment]  # Get the emoji directly from the mapping without a default value
    
    return render_template('result.html', comment=comment, sentiment=sentiment, emoji=emoji)

if __name__ == '__main__':
    app.run(debug=True)