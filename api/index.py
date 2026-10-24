from flask import Flask, render_template, request
import numpy as np
import joblib
import os
import tensorflow as tf

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Path ke model dan file pendukung
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'DeepResNet_HeartDisease.h5')
SCALER_PATH = os.path.join(BASE_DIR, '..', 'scaler.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, '..', 'label_encoders.pkl')

# Load model dan file pendukung
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoders = joblib.load(ENCODER_PATH)

feature_names = [
    'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth',
    'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory', 'Diabetic',
    'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma',
    'KidneyDisease', 'SkinCancer'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        input_data = []

        for feature in feature_names:
            value = data[feature]
            if feature in label_encoders:
                value = label_encoders[feature].transform([value])[0]
            else:
                value = float(value)
            input_data.append(value)

        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        input_reshaped = input_scaled.reshape(1, len(feature_names), 1)

        prob = model.predict(input_reshaped)[0][0]
        result = "Yes" if prob > 0.5 else "No"
        risk = f"{prob * 100:.2f}%"

        return render_template('index.html',
                               prediction=result,
                               probability=risk,
                               data=data)

    except Exception as e:
        return render_template('index.html', error=str(e))

# WSGI entry point untuk Vercel
def handler(request, response):
    return app(request.environ, response)