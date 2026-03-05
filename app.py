from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model/model.pkl','rb'))

@app.route('/')
def home():
    return "Insurance Fraud Detection Model is Running"

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    if prediction[0] == 1:
        return "Fraud Detected"
    else:
        return "No Fraud"

if __name__ == "__main__":
    app.run(debug=True)
