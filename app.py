from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

model = tf.keras.models.load_model('LSTM_Final_96.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news']
        seq = tokenizer.texts_to_sequences([news_text])
        padded = pad_sequences(seq, maxlen=300, padding='post')
        
        prediction = model.predict(padded)[0][0]
        result = "Real News " if prediction < 0.5 else "Fake News "
        confidence = round(float(prediction if prediction > 0.5 else 1-prediction) * 100, 2)
        
        return render_template('index.html', prediction_text=result, confidence_text=f"Confidence: {confidence}%")

if __name__ == "__main__":
    app.run(debug=True)