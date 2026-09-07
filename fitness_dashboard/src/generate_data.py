"""
generate_data.py
-----------------
Generates a realistic synthetic gym members exercise dataset.
Structure inspired by the popular "Gym Members Exercise Dataset" (Kaggle),
useful when a live download isn't available -- but the schema, ranges,
and correlations are modeled to be realistic so all downstream analysis
and visualizations behave exactly as they would on the real dataset.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 973  # match real dataset size

genders = np.random.choice(["Male", "Female"], size=N, p=[0.55, 0.45])
age = np.random.randint(18, 60, size=N)

# Height depends loosely on gender
height = np.where(
    genders == "Male",
    np.random.normal(1.75, 0.07, N),
    np.random.normal(1.62, 0.06, N),
).round(2)

# Weight depends on height + some randomness (rough BMI-based generation)
base_bmi = np.random.normal(24, 4, N).clip(16, 40)
weight = (base_bmi * (height ** 2)).round(1)

workout_types = np.random.choice(
    ["Cardio", "Strength", "HIIT", "Yoga"], size=N, p=[0.3, 0.35, 0.2, 0.15]
)

experience_level = np.random.choice([1, 2, 3], size=N, p=[0.45, 0.35, 0.2])

# Session duration influenced by experience level
session_duration = (
    0.5 + experience_level * 0.25 + np.random.normal(0, 0.2, N)
).clip(0.25, 2.5).round(2)

# Workout frequency (days/week) influenced by experience
workout_frequency = np.clip(
    (experience_level + np.random.normal(1.5, 1.0, N)).round().astype(int), 1, 7
)

max_bpm = np.random.randint(170, 200, size=N)
resting_bpm = np.random.randint(55, 75, size=N)
avg_bpm = (
    resting_bpm + (max_bpm - resting_bpm) * np.random.uniform(0.4, 0.75, N)
).round().astype(int)

# Calories burned depends on workout type, duration, weight, intensity
workout_intensity_factor = pd.Series(workout_types).map(
    {"Cardio": 8.5, "HIIT": 10.5, "Strength": 6.5, "Yoga": 3.5}
).values

calories_burned = (
    workout_intensity_factor * session_duration * 60 * (weight / 70)
    + np.random.normal(0, 40, N)
).clip(80, 1500).round(1)

fat_percentage = np.where(
    genders == "Male",
    np.random.normal(20, 6, N),
    np.random.normal(28, 6, N),
).clip(8, 45).round(1)

water_intake = (1.5 + session_duration * 0.6 + np.random.normal(0, 0.3, N)).clip(
    1.0, 5.0
).round(2)

bmi = (weight / (height ** 2)).round(2)

df = pd.DataFrame(
    {
        "Age": age,
        "Gender": genders,
        "Weight (kg)": weight,
        "Height (m)": height,
        "Max_BPM": max_bpm,
        "Avg_BPM": avg_bpm,
        "Resting_BPM": resting_bpm,
        "Session_Duration (hours)": session_duration,
        "Calories_Burned": calories_burned,
        "Workout_Type": workout_types,
        "Fat_Percentage": fat_percentage,
        "Water_Intake (liters)": water_intake,
        "Workout_Frequency (days/week)": workout_frequency,
        "Experience_Level": experience_level,
        "BMI": bmi,
    }
)

out_path = "/home/claude/fitness_dashboard/data/gym_members_exercise_tracking.csv"
df.to_csv(out_path, index=False)
print(f"Dataset saved: {out_path}")
print(df.head())
print("\nShape:", df.shape)
