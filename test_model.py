import keras.models
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np

model = keras.models.load_model("model.h5")
tokenizer = Tokenizer(num_words=2000, split=' ')
model.summary()
text = input("Enter your review here:")
text = [text]
# vectorizing the tweet by the pre-fitted tokenizer instance
tokenizer.fit_on_texts(text)
X = tokenizer.texts_to_sequences(text)
# padding the tweet to have exactly the same shape as `embedding_2` input
X = pad_sequences(X, maxlen=28, dtype='int32', value=0)
print(X)
sentiment = model.predict(X)[0]
if np.argmax(sentiment) == 0:
    print("negative")
elif np.argmax(sentiment) == 1:
    print("positive")
