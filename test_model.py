# ignore all the gpu related warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# importing some of the requirements
import keras.models
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
# loading the trained model and applying text pre-processing
model = keras.models.load_model("model.h5")
tokenizer = Tokenizer(num_words=2000, split=' ')
# taking input from the user
text = input("Enter your review here:")
text = [text]
# vectorizing the review by the pre-fitted tokenizer instance
tokenizer.fit_on_texts(text)
X = tokenizer.texts_to_sequences(text)
# padding the review to have exactly the same shape as `embedding_2` input
X = pad_sequences(X, maxlen=28, dtype='int32', value=0)
print(X)
sentiment = model.predict(X)[0]
if np.argmax(sentiment) == 0:
    print("negative")
elif np.argmax(sentiment) == 1:
    print("positive")
