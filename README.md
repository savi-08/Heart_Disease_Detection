# 🩺 Heart Disease Detection using Machine Learning

This project predicts whether a patient has heart disease using various medical attributes. It includes data exploration, model training, evaluation, and a CLI-based prediction tool.

## 📁 Project Structure

Heart_Disease_Detection/
├── dataset/
│ └── heart.csv
├── models/
│ └── random_forest_model.pkl
├── notebooks/
│ ├── heart_disease_analysis.py
│ ├── train_models.py
│ └── predict_cli.py
├── requirements.txt
├── README.md
└── .gitignore

markdown
Copy
Edit

## 🧠 Dataset Description

Each row contains information about a patient including:

- Age, Sex, Chest Pain Type, Blood Pressure, Cholesterol
- Fasting Blood Sugar, ECG Results, Max Heart Rate, Exercise Angina
- ST Depression (Oldpeak), ST Slope
- **Target:** `0 = Normal`, `1 = Heart Disease`

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Heart_Disease_Detection.git
cd Heart_Disease_Detection
```
### 2. Create virtual environment and install dependencies
```bash
python -m venv venv
.\venv\Scripts\activate      
pip install -r requirements.txt
```
### 3. Run CLI-based prediction
```bash
python notebooks/predict_cli.py
```
## ⚙️ Models Used
Logistic Regression (baseline)

Random Forest (final model)

## 📊 Tools & Libraries
Python, Pandas, NumPy, Scikit-Learn

Matplotlib, Seaborn

Joblib

## 💡 Future Improvements
Build a web app using Streamlit

Add more models and compare

Export predictions to CSV
