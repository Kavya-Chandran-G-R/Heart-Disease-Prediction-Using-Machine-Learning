# -*- coding: utf-8 -*-

import numpy as np
import joblib
from flask import Flask, request, render_template
import pandas as pd

# Load ML model
scaler_path = r'transform.sav'
sc = joblib.load(scaler_path)

model_path = r'model.sav'
model = joblib.load(model_path)

heart_df = pd.read_csv('heart.csv')
X = heart_df.drop('target',axis='columns')
sc.fit_transform(X)

# Create application
app = Flask(__name__)

# Bind home function to URL
@app.route('/')
def home():
    return render_template('Heart Disease Classifier.html')

# Bind predict function to URL
@app.route('/predict', methods =['POST'])
def predict():
    
    # Put all form entries values in a list 
    features = [float(i) for i in request.form.values()]
    # Convert features to array
    array_features = [np.array(features)]
    print(array_features)
    # Predict features
    prediction = model.predict(sc.transform(array_features))
    print(sc.transform(array_features))
    
    
    output = prediction
    print(output)
    
    # Check the output values and retrive the result with html tag based on the value
    if output == 0:
        return render_template('Heart Disease Classifier.html', 
                               result1 = 'The patient is not likely to have heart disease!')
    if output == 1:
        return render_template('Heart Disease Classifier.html', 
                               result2 = 'The patient is likely to have heart disease!')

if __name__ == '__main__':
#Run the application
    app.run()

    
    
    
