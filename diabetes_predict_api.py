from flask import Flask, request, jsonify
import joblib
import numpy as np
import requests
import gzip
import shutil

app = Flask(__name__)

# Function to download and extract model from Google Drive
def download_and_load_model():
    file_url = "https://drive.google.com/uc?id=16NGlvsASG2atOC2QfRui4bo068Jo190G"  # Google Drive file ID
    model_path = "diabetes_model_fixed.joblib"

    # Download the model
    response = requests.get(file_url, stream=True)
    with open(model_path, "wb") as file:
        shutil.copyfileobj(response.raw, file)
    print("Model has been downloaded")

    # Load the model
    return joblib.load(model_path)

# Load the model
model = download_and_load_model()

@app.route('/')
def home():
    return "Diabetes Prediction API is Running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.get_json()

        # Ensure the correct data format
        if not data or 'features' not in data:
            return jsonify({"error": "Invalid input. Please provide 'features' in JSON format."}), 400

        # Convert input to numpy array
        features = np.array(data['features']).reshape(1, -1)

        # Make a prediction
        prediction = model.predict(features)

        # Return the prediction result
        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
