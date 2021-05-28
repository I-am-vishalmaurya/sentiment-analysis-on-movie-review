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
    return render_template('index.html', display_mode = "none")

# route for prediction of the review
@app.route('/predict', methods=['POST'])
def predict():
    global model_inputs
    global model_outputs
    # get text from the incoming requests
    text = request.form['input_text']
    #convert text into model input vector
    final_feature = input_transformer.transform([text])

    #use classifier to predict the sentiment of review
    prediction = model.predict(final_feature)

    #store model input and output
    model_inputs = text
    model_outputs = prediction[0]
    return model_outputs

# route for incremental trianing
@app.route('/save_pred', methods=['POST'])
def save_pred():
    # retrieve global variable
    global model_inputs
    global model_outputs

    # vectorize user inputs
    final_features = input_transformer.transform([model_inputs])

    # get user button choice --> correct ot incorrect
    save_type = request.form['save_type']

    #return text
    return_text = "Thank you for teaching me!!"

    #modify global variable if user selected incorrect for retraining
    if(save_type == 'incorrect'):
        return_text = 'Thank you for strenghtening me!!'
        if(model_outputs =='p'):
            model_outputs == 'n'
        elif(model_outputs == 'n'):
            model_outputs == 'p'
        else:
            print("Error: Model output is neither N or P")

    # strengthen weight for particular connection
    max_iter = 100
    counter = 0
    for i in range(0, max_iter):
        model.patial_fit(final_features , [model_outputs])
        if(model.predict(final_features) == [model_outputs]):
            counter = i
            break

    # save trained model pickle
    joblib.dump(model, (app.static_folder + 'models/input_transformer.pkl'))

    # fiels inside csv to store and retrieve for retrain verification
    fields = [model_inputs, model_outputs, counter]

    # retrain model
    with open((app.root_path + '/user_teaching.csv'), 'a') as file:
        writer = csv.writer(file)
        writer.writerow(fields)

    #return confirmation code for user
    return return_text

if __name__ == "__main__":
    app.run(debug=True)