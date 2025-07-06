import joblib
import numpy as np
import pandas as pd

model = joblib.load('models/random_forest_model.pkl')

columns = ['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol',
           'fasting blood sugar', 'resting ecg', 'max heart rate',
           'exercise angina', 'oldpeak', 'ST slope']

sample_patient = pd.DataFrame([[62, 1, 4, 140, 268, 0, 2, 160, 0, 3.6, 2]], columns=columns) #Has heart disease
#sample_patient = pd.DataFrame([[45, 0, 2, 120, 200, 0, 0, 160, 0, 0.5, 1]],columns=columns) #Is healthy
prediction = model.predict(sample_patient)[0]

if prediction == 1:
    print("💔 Patient is likely to have heart disease.")
else:
    print("💖 Patient is likely healthy (no heart disease).")
