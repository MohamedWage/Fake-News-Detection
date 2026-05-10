from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import joblib
import numpy as np
import re
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('punkt_tab')
en_stop = set(nltk.corpus.stopwords.words('english'))

app = Flask(__name__)

try:
    lstm_w2v_model = tf.keras.models.load_model('LSTM_Final.h5')
    lstm_tfidf_model = tf.keras.models.load_model('LSTM_TFIDF_Model.h5')
    lr_tfidf_model = joblib.load('lr_model.pkl')
    lr_w2v_model = joblib.load('lr_w2v_model.pkl')
    
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    tfidf_vec = joblib.load('tfidf_vectorizer.pkl')
    tfidf_vec_lstm = joblib.load('tfidf_vectorizer_lstm.pkl')
    w2v_model = Word2Vec.load('word2vec_model.model')
    print("All models and vectorizers loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    tokens = word_tokenize(text)
    clean_tokens = [word for word in tokens if word not in en_stop and len(word) > 2]
    return " ".join(clean_tokens)

def get_w2v_vector(text, model):
    words = text.split()
    vectors = [model.wv[w] for w in words if w in model.wv]
    if not vectors: return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

def extract_tfidf_reasons(text, vectorizer):
    matrix = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = matrix.toarray().flatten()
    top_indices = scores.argsort()[-3:][::-1]
    res = [feature_names[i] for i in top_indices if scores[i] > 0]
    return res if res else ["statistical pattern"]


def extract_w2v_reasons(text, model_wv=None, is_lr=False):
    raw_words = text.lower().split()
    if is_lr and model_wv:
        words = [w for w in raw_words if w in model_wv and len(w) > 2]
    else:
        words = [w for w in raw_words if len(w) > 2]

    if not words:
        words = [w for w in raw_words if len(w) > 2][:3]
        if not words: return ["context analysis"]
    word_importance = {}
    for w in words:
        index = tokenizer.word_index.get(w, 100000) 
        word_importance[w] = index
    sorted_words = sorted(word_importance.items(), key=lambda x: x[1])
    
    return [w[0] for w in sorted_words[:3]]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news']
    choice = request.form['model_choice']
    
    # التنظيف المطابق للـ Jupyter
    cleaned = clean_text(news_text)
    
    result, conf, m_name, reasons_list = "", 0.0, "", []

    if choice == 'lstm_w2v':
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=300, padding='post')
        pred = lstm_w2v_model.predict(padded)[0][0]
        result = "Real News" if pred > 0.5 else "Fake News"
        conf = pred if pred > 0.5 else 1 - pred
        m_name = "LSTM + Word2Vec"
        reasons_list = extract_w2v_reasons(cleaned, tokenizer,is_lr=False)

    elif choice == 'lr_tfidf':
        vec = tfidf_vec.transform([cleaned])
        pred = lr_tfidf_model.predict(vec)[0]
        prob = lr_tfidf_model.predict_proba(vec)[0]
        result = "Real News" if pred == 1 else "Fake News"
        conf = max(prob)
        m_name = "Logistic Regression + TF-IDF"
        reasons_list = extract_tfidf_reasons(cleaned, tfidf_vec)

    elif choice == 'lr_w2v':
        vec = get_w2v_vector(cleaned, w2v_model) # Mean vector
        pred = lr_w2v_model.predict([vec])[0]
        prob = lr_w2v_model.predict_proba([vec])[0]
        result = "Real News" if pred == 1 else "Fake News"
        conf = max(prob)
        m_name = "Logistic Regression + Word2Vec"
        # أسباب مبنية على منطق الـ LR (Vector Magnitude)
        reasons_list = extract_w2v_reasons(cleaned, w2v_model.wv,is_lr=True)

    elif choice == 'lstm_tfidf':
        vec = tfidf_vec_lstm.transform([cleaned]).toarray() 
        if vec.shape[1] == 3000:
            vec_reshaped = vec.reshape(vec.shape[0], 50, 60)
            pred = lstm_tfidf_model.predict(vec_reshaped)[0][0]
            result = "Real News" if pred > 0.5 else "Fake News"
            conf = pred if pred > 0.5 else 1 - pred
            # استخدام الفيكتورايزر الخاص بالـ LSTM (3000 features)
            reasons_list = extract_tfidf_reasons(cleaned, tfidf_vec_lstm)
        else:
            return f"Error: Vectorizer size is {vec.shape[1]}, but Model needs 3000."
        m_name = "LSTM + TF-IDF"


    reason_str = ", ".join([f'"{r}"' for r in reasons_list])

    return render_template('index.html', 
                           prediction_text=result, 
                           reason=reason_str,
                           confidence_text=f"Confidence: {round(conf*100, 2)}%",
                           model_used=m_name)

if __name__ == "__main__":
    app.run(debug=True)