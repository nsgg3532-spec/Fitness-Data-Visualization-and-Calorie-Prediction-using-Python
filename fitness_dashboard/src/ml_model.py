"""
ml_model.py
------------
Trains a simple Linear Regression model to predict Calories_Burned
from other fitness features, and saves the trained model to disk.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

DATA_PATH = "data/gym_members_exercise_tracking.csv"
MODEL_PATH = "src/calories_model.pkl"
ENCODER_PATH = "src/workout_encoder.pkl"

df = pd.read_csv(DATA_PATH)

# Encode categorical column (Workout_Type) into numbers
le = LabelEncoder()
df["Workout_Type_Encoded"] = le.fit_transform(df["Workout_Type"])

# Features used to predict calories burned
features = [
    "Age", "Weight (kg)", "Height (m)", "Session_Duration (hours)",
    "Avg_BPM", "Workout_Type_Encoded", "Experience_Level"
]
X = df[features]
y = df["Calories_Burned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f} kcal")

joblib.dump(model, MODEL_PATH)
joblib.dump(le, ENCODER_PATH)
print(f"\nModel saved to {MODEL_PATH}")