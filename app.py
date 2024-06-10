from flask import Flask, render_template, request, url_for, redirect
import numpy as np
import re
import os
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.layers import LSTM

class CustomLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)  # Remove 'time_major' if it exists
        super().__init__(*args, **kwargs)


custom_objects = {
    'Orthogonal': Orthogonal,
    'LSTM': CustomLSTM
}

IMAGE_FOLDER = os.path.join('static', 'img_pool')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

def init():
    global model,graph
    model = load_model('sentiment_analysis_model_new.h5', custom_objects=custom_objects)
    # graph = tf.get_default_graph()

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("home.html")

@app.route('/sentiment_analysis_prediction', methods=['POST', 'GET'])
def sent_anly_prediction():
    if request.method == 'POST':
        text = request.form['text']
        sentiment = ''
        max_review_length = 500
        word_to_id = imdb.get_word_index()
        strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
        text = text.lower().replace("<br />", " ")
        text = re.sub(strip_special_chars, "", text.lower())

        words = text.split()
        x_test = [[word_to_id[word] if (word in word_to_id and word_to_id[word] <= 5000) else 0 for word in words]]
        x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)
        vector = np.array([x_test.flatten()])

        # with graph.as_default():
        probability = model.predict(np.array([vector][0]))[0][0]
        class1 = np.argmax(probability, axis=-1)

        if class1 == 0:
            sentiment = 'Negative'
            # img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Sad_Emoji.png')
        else:
            sentiment = 'Positive'
            # img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Smiling_Emoji.png')

        return render_template('home.html', text=text, sentiment=sentiment, probability=probability)
    return render_template('home.html')

if __name__ == "__main__":
    init()
    app.run(debug=True, port=8080)
