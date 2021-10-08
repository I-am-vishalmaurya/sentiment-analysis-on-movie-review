from flask import Flask, request, render_template
import joblib
import csv

# app initialization
app = Flask(__name__)

# load model
input_transformer = joblib.load(open(app.static_folder + '/models/input_transformer.pkl', 'rb'))
model =  joblib.load(open(app.static_folder + '/models/review_sentiment.pkl', 'rb'))

# gloabl variable for data persistence across request
model_inputs = ""
model_outputs = ""

# main index page route
@app.route('/')
def home():
    return render_template('index.html')

# route for prediction of the review
@app.route('/predict', methods=['POST'])
def predict():
    # get text from the incoming requests
    text = request.form['message']
    #convert text into model input vector
    final_feature = input_transformer.transform([text])

    #use classifier to predict the sentiment of review
    my_prediction = model.predict(final_feature)

    return render_template('result.html', prediction = my_prediction)


if __name__ == "__main__":
    app.run(debug=True)