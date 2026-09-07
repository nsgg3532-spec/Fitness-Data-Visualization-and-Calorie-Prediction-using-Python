"""
eda_visualizations.py
----------------------
Performs exploratory data analysis (EDA) on the gym members dataset
and generates a set of static charts using matplotlib and seaborn.
Charts are saved to outputs/charts/ as PNG files.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="viridis")

DATA_PATH = "/home/claude/fitness_dashboard/data/gym_members_exercise_tracking.csv"
OUT_DIR = "/home/claude/fitness_dashboard/outputs/charts"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(df.info())
print("\nSummary statistics:\n", df.describe())
print("\nMissing values:\n", df.isnull().sum())

# ---------------------------------------------------------------
# 1. Age distribution
# ---------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.histplot(df["Age"], bins=20, kde=True, color="teal")
plt.title("Age Distribution of Gym Members")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_age_distribution.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 2. Average calories burned by workout type
# ---------------------------------------------------------------
plt.figure(figsize=(8, 5))
avg_cal = df.groupby("Workout_Type")["Calories_Burned"].mean().sort_values(ascending=False)
sns.barplot(x=avg_cal.index, y=avg_cal.values, hue=avg_cal.index, legend=False, palette="mako")
plt.title("Average Calories Burned by Workout Type")
plt.xlabel("Workout Type")
plt.ylabel("Avg Calories Burned")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_calories_by_workout_type.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 3. Session duration vs calories burned (scatter)
# ---------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df, x="Session_Duration (hours)", y="Calories_Burned",
    hue="Workout_Type", alpha=0.7
)
plt.title("Session Duration vs Calories Burned")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_duration_vs_calories.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 4. Correlation heatmap
# ---------------------------------------------------------------
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include="number")
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Fitness Metrics")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_correlation_heatmap.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 5. Fat percentage by gender (boxplot)
# ---------------------------------------------------------------
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Gender", y="Fat_Percentage", hue="Gender", legend=False, palette="Set2")
plt.title("Fat Percentage Distribution by Gender")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_fat_percentage_by_gender.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 6. Workout frequency vs experience level
# ---------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df, x="Experience_Level", y="Workout_Frequency (days/week)",
    hue="Experience_Level", legend=False, palette="crest"
)
plt.title("Workout Frequency by Experience Level")
plt.xlabel("Experience Level (1=Beginner, 3=Advanced)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/06_frequency_by_experience.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 7. BMI distribution by workout type (violin)
# ---------------------------------------------------------------
plt.figure(figsize=(9, 5))
sns.violinplot(data=df, x="Workout_Type", y="BMI", hue="Workout_Type", legend=False, palette="flare")
plt.title("BMI Distribution Across Workout Types")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/07_bmi_by_workout_type.png", dpi=150)
plt.close()

print(f"\nAll charts saved to {OUT_DIR}")

# ---------------------------------------------------------------
# Key insights (printed to console + saved to a text file)
# ---------------------------------------------------------------
insights = []
top_workout = avg_cal.idxmax()
insights.append(
    f"1. '{top_workout}' burns the most calories on average "
    f"({avg_cal.max():.1f} kcal/session)."
)
corr_dur_cal = df["Session_Duration (hours)"].corr(df["Calories_Burned"])
insights.append(
    f"2. Session duration and calories burned are strongly correlated (r={corr_dur_cal:.2f})."
)
avg_freq_by_exp = df.groupby("Experience_Level")["Workout_Frequency (days/week)"].mean()
insights.append(
    f"3. Advanced members (level 3) work out {avg_freq_by_exp[3]:.1f} days/week on average, "
    f"vs {avg_freq_by_exp[1]:.1f} days/week for beginners."
)
fat_by_gender = df.groupby("Gender")["Fat_Percentage"].mean()
insights.append(
    f"4. Average fat percentage: Male {fat_by_gender.get('Male', 0):.1f}%, "
    f"Female {fat_by_gender.get('Female', 0):.1f}%."
)

with open(f"{OUT_DIR}/../insights.txt", "w") as f:
    f.write("KEY INSIGHTS FROM FITNESS DATA ANALYSIS\n")
    f.write("=" * 45 + "\n\n")
    for line in insights:
        f.write(line + "\n")

print("\n".join(insights))
