import os
os.environ['TF_CPP_MIN__LOG_LEVEL'] = '3'

import numpy as np
from flask import *
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

model = load_model('model.h5')
tokenizer = Tokenizer(num_words=2000, split=' ')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        messages = request.form['message']
        tokenizer.fit_on_texts(messages)
        token = tokenizer.texts_to_sequences(messages)
        message = pad_sequences(token, maxlen=28, dtype='int32', value=0)
        sentiment = model.predict(message)[0]
        if np.argmax(sentiment) == 0:
            return render_template('index.html', prediction='Its a negative review')
        elif np.argmax(sentiment) == 1:
            return render_template('index.html', prediction='Great!! this is a poitive review')



if __name__ == "__main__":
    app.run(debug=True)
