from flask import Flask, render_template, request, flash
import numpy as np
import pandas as pd
from model_loader import load_models
import logging

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for flash messages

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    preprocessor, encoder, classifier = load_models()
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    preprocessor, encoder, classifier = None, None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not all([preprocessor, encoder, classifier]):
        flash("Model loading failed. Please try again later.", "error")
        return render_template('index.html')
    
    try:
        input_data = {
            'gender': request.form['gender'],
            'SeniorCitizen': int(request.form['SeniorCitizen']),
            'Partner': request.form['Partner'],
            'Dependents': request.form['Dependents'],
            'tenure': float(request.form['tenure']),
            'PhoneService': request.form['PhoneService'],
            'MultipleLines': request.form['MultipleLines'],
            'InternetService': request.form['InternetService'],
            'OnlineSecurity': request.form['OnlineSecurity'],
            'OnlineBackup': request.form['OnlineBackup'],
            'DeviceProtection': request.form['DeviceProtection'],
            'TechSupport': request.form['TechSupport'],
            'StreamingTV': request.form['StreamingTV'],
            'StreamingMovies': request.form['StreamingMovies'],
            'Contract': request.form['Contract'],
            'PaperlessBilling': request.form['PaperlessBilling'],
            'PaymentMethod': request.form['PaymentMethod'],
            'MonthlyCharges': float(request.form['MonthlyCharges']),
            'TotalCharges': float(request.form['TotalCharges'])
        }

        df = pd.DataFrame([input_data])
        processed = preprocessor.transform(df)
        encoded = encoder.predict(processed)
        pred_proba = classifier.predict_proba(encoded)[0][1]
        prediction = 'Yes' if pred_proba > 0.5 else 'No'
        
        result = {
            'prediction': prediction,
            'probability': round(pred_proba * 100, 2),
            'details': input_data
        }
        
        return render_template('index.html', result=result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        flash("An error occurred during prediction. Please check your inputs.", "error")
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)