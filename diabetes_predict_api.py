from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import pickle
import joblib
import numpy as np

# Load the model
diabetes_model = pickle.load(open('diabetes_model_fixed.joblib', 'rb'))

app = FastAPI()

# HTML content for the frontend
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Diabetes Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f9;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .container {
            background: #ffffff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            text-align: center;
            width: 100%;
            max-width: 400px;
        }
        h1 {
            color: #333;
            font-size: 1.5em;
            margin-bottom: 20px;
        }
        form {
            display: flex;
            flex-direction: column;
        }
        input {
            margin-bottom: 15px;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 1em;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 1em;
        }
        button:hover {
            background-color: #45a049;
        }
        #result {
            margin-top: 20px;
            font-size: 1.2em;
            color: #333;
        }
        #result.positive {
            color: red; /* Warna untuk prediksi positif */
        }
        #result.negative {
            color: green; /* Warna untuk prediksi negatif */
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Diabetes Prediction</h1>
        <form id="prediction-form">
            <input type="number" name="Pregnancies" placeholder="Jumlah Kehamilan (Pregnancies)" required>
            <input type="number" name="Glucose" placeholder="Glukosa (Glucose)" required>
            <input type="number" name="BloodPressure" placeholder="Tekanan Darah (Blood Pressure)" required>
            <input type="number" name="SkinThickness" placeholder="Ketebalan Kulit (Skin Thickness)" required>
            <input type="number" name="Insulin" placeholder="Insulin" required>
            <input type="number" name="BMI" placeholder="Indeks Massa Tubuh (BMI)" step="any" required>
            <input type="number" name="DiabetesPedigreeFunction" placeholder="Diabetes Pedigree Function" step="any" required>
            <input type="number" name="Age" placeholder="Usia (Age)" required>
            <button type="submit">Prediksi</button>
        </form>
        <div id="result"></div>
    </div>
    <script>
        document.getElementById('prediction-form').onsubmit = async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            const resultDiv = document.getElementById('result');

            if (result.error) {
                resultDiv.innerHTML = `<h2>Error: ${result.error}</h2>`;
                resultDiv.className = ""; // Hapus kelas gaya jika ada error
            } else {
                resultDiv.innerHTML = `<h2>${result.prediction}</h2>`;
                if (result.prediction.includes("Positif")) {
                    resultDiv.className = "positive"; // Tambahkan kelas untuk warna merah
                } else {
                    resultDiv.className = "negative"; // Tambahkan kelas untuk warna hijau
                }
            }
        };
    </script>
</body>
</html>
"""

@app.get('/', response_class=HTMLResponse)
async def read_root():
    return html_content

@app.post('/predict')
async def predict(
    Pregnancies: float = Form(...),
    Glucose: float = Form(...),
    BloodPressure: float = Form(...),
    SkinThickness: float = Form(...),
    Insulin: float = Form(...),
    BMI: float = Form(...),
    DiabetesPedigreeFunction: float = Form(...),
    Age: float = Form(...),
):
    try:
        # Validasi input DiabetesPedigreeFunction
        if not (0 <= DiabetesPedigreeFunction <= 2):
            return {"error": "Diabetes Pedigree Function harus bernilai antara 0 dan 2"}

        # Validasi input BMI
        if not (10 <= BMI <= 50):
            return {"error": "BMI harus bernilai antara 10 dan 50"}

        # Prepare the input for the model
        input_data = np.array([[
            Pregnancies, Glucose, BloodPressure, SkinThickness,
            Insulin, BMI, DiabetesPedigreeFunction, Age
        ]])

        # Make prediction
        prediction = diabetes_model.predict(input_data)

        # Interpret the result
        result = "Positif Diabetes" if prediction[0] == 1 else "Negatif Diabetes"
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}
