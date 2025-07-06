import joblib
import pandas as pd

model = joblib.load('models/random_forest_model.pkl')

columns = ['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol',
           'fasting blood sugar', 'resting ecg', 'max heart rate',
           'exercise angina', 'oldpeak', 'ST slope']

def get_valid_input(prompt, valid_range, cast_func=int):
    while True:
        try:
            value = cast_func(input(prompt))
            if value in valid_range:
                return value
            else:
                print(f"❌ Please enter a value between {min(valid_range)} and {max(valid_range)}")
        except ValueError:
            print("❌ Invalid input. Please enter a valid number.")

print("\n🩺 Heart Disease Prediction - Enter Patient Info Below\n")

# Input
values = [
    get_valid_input("Age: ", range(1, 130)),
    get_valid_input("Sex (0 = Female, 1 = Male): ", [0, 1]),
    get_valid_input("Chest Pain Type (1=typical angina, 2=atypical angina, 3=non-anginal pain, 4=asymptomatic): ", [1, 2, 3, 4]),
    get_valid_input("Resting BP (mm Hg): ", range(70, 251)),
    get_valid_input("Cholesterol (mg/dl): ", range(100, 601)),
    get_valid_input("Fasting Blood Sugar (0 = sugar<120mg/dL, 1 = sugar>120mg/dL): ", [0, 1]),
    get_valid_input("Resting ECG (0=normal, 1=abnormal, 2=LV hypertrophy): ", [0, 1, 2]),
    get_valid_input("Max Heart Rate Achieved: ", range(71, 202)),
    get_valid_input("Exercise Induced Angina (0 = No, 1 = Yes): ", [0, 1])
]

while True:
    try:
        oldpeak = float(input("Oldpeak (ST depression): "))
        if 0.0 <= oldpeak <= 10.0:
            values.append(oldpeak)
            break
        else:
            print("❌ Enter a value between 0.0 and 10.0")
    except ValueError:
        print("❌ Invalid input. Enter a number.")

values.append(get_valid_input("ST Slope (1 = upward, 2 = flat, 3 = downward): ", [1, 2, 3]))

# Prediction
sample = pd.DataFrame([values], columns=columns)
prediction = model.predict(sample)[0]
print(f"\nPredicted class: {prediction} (0 = Normal, 1 = Heart Disease)")

# Output
print("\n💔 Patient is likely to have heart disease." if prediction == 1
      else "\n💖 Patient is likely healthy (no heart disease).")
