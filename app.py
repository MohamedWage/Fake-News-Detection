from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import joblib
import numpy as np
import re
from gensim.models import Word2Vec

app = Flask(__name__)

# --- Load all components ---
try:
    # Models
    lstm_w2v_model = tf.keras.models.load_model('LSTM_Final.h5')
    lstm_tfidf_model = tf.keras.models.load_model('LSTM_TFIDF_Model.h5')
    lr_tfidf_model = joblib.load('lr_model.pkl')
    lr_w2v_model = joblib.load('lr_w2v_model.pkl')
    
    # Preprocessors
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    tfidf_vec = joblib.load('tfidf_vectorizer.pkl')
    tfidf_vec_lstm = joblib.load('tfidf_vectorizer_lstm.pkl')
    w2v_model = Word2Vec.load('word2vec_model.model')
    print("All models and vectorizers loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

def get_w2v_vector(text, model):
    words = text.split()
    vectors = [model.wv[w] for w in words if w in model.wv]
    if not vectors: return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news']
    choice = request.form['model_choice']
    cleaned = clean_text(news_text)
    
    result, conf, m_name = "", 0.0, ""

    if choice == 'lstm_w2v':
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=300, padding='post')
        pred = lstm_w2v_model.predict(padded)[0][0]
        result = "Real News" if pred > 0.5 else "Fake News"
        conf = pred if pred > 0.5 else 1 - pred
        m_name = "LSTM + Word2Vec"

    elif choice == 'lr_tfidf':
        vec = tfidf_vec.transform([cleaned])
        pred = lr_tfidf_model.predict(vec)[0]
        prob = lr_tfidf_model.predict_proba(vec)[0]
        result = "Real News" if pred == 1 else "Fake News"
        conf = max(prob)
        m_name = "Logistic Regression + TF-IDF"

    elif choice == 'lr_w2v':
        vec = get_w2v_vector(cleaned, w2v_model)
        pred = lr_w2v_model.predict([vec])[0]
        prob = lr_w2v_model.predict_proba([vec])[0]
        result = "Real News" if pred == 1 else "Fake News"
        conf = max(prob)
        m_name = "Logistic Regression + Word2Vec"

    elif choice == 'lstm_tfidf':
        vec = tfidf_vec_lstm.transform([cleaned]).toarray() 
        
        if vec.shape[1] == 3000:
            vec_reshaped = vec.reshape(vec.shape[0], 50, 60)
            pred = lstm_tfidf_model.predict(vec_reshaped)[0][0]
            result = "Real News" if pred > 0.5 else "Fake News"
            conf = pred if pred > 0.5 else 1 - pred
        else:
            return f"Error: Vectorizer size is {vec.shape[1]}, but Model needs 3000. Re-save your vectorizer from Jupyter."
        
        m_name = "LSTM + TF-IDF"

    return render_template('index.html', 
                           prediction_text=f"{result} Detected", 
                           confidence_text=f"Confidence: {round(conf*100, 2)}%",
                           model_used=m_name)

if __name__ == "__main__":
    app.run(debug=True)