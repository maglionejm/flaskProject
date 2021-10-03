import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/predict", methods = ["POST"])
def predict():
    json =request.json
    query_df = pd.DataFrame(json)
    prediction = model.predict(query_df)
    return jsonify({"Prediction": list(prediction)})

if __name__ == "__main__":
    flask_app.run(debug=True)