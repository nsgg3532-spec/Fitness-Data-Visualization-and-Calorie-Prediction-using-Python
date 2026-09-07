# 💪 Fitness Analytics — Data Visualization Project

A complete data visualization project analyzing gym members' exercise patterns,
built with **Python, Pandas, Matplotlib, Seaborn, Plotly, and Streamlit**.

## 📌 Project Overview

This project analyzes a fitness/gym dataset (973 members, 15 attributes) to uncover
patterns in workout habits, calorie burn, heart rate, body composition, and
experience level. It includes both a **static EDA report** (matplotlib/seaborn)
and a **live interactive dashboard** (Plotly + Streamlit) with filters.

## 🗂️ Project Structure

```
fitness_dashboard/
├── data/
│   └── gym_members_exercise_tracking.csv   # Dataset (973 rows x 15 columns)
├── src/
│   ├── generate_data.py                    # Dataset generation script
│   └── eda_visualizations.py               # Static EDA + chart generation
├── outputs/
│   ├── charts/                             # 7 static PNG charts
│   └── insights.txt                        # Key findings, auto-generated
├── app.py                                  # Interactive Streamlit dashboard
├── requirements.txt
└── README.md
```

## 📊 Dataset Columns

| Column | Description |
|---|---|
| Age, Gender | Demographics |
| Weight (kg), Height (m), BMI | Body metrics |
| Max_BPM, Avg_BPM, Resting_BPM | Heart rate metrics |
| Session_Duration (hours) | Workout length |
| Calories_Burned | Calories burned per session |
| Workout_Type | Cardio / Strength / HIIT / Yoga |
| Fat_Percentage | Body fat % |
| Water_Intake (liters) | Daily water intake |
| Workout_Frequency (days/week) | How often they train |
| Experience_Level | 1 (Beginner) to 3 (Advanced) |

> **Note:** Data is realistically simulated to mirror the structure and
> statistical relationships of the popular "Gym Members Exercise Dataset"
> (Kaggle). To use real data instead, replace
> `data/gym_members_exercise_tracking.csv` with a downloaded dataset that has
> matching column names, and re-run the scripts.

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Regenerate the dataset
```bash
python src/generate_data.py
```

### 3. Run static EDA (generates charts + insights)
```bash
python src/eda_visualizations.py
```
Charts will be saved to `outputs/charts/`.

### 4. Launch the interactive dashboard
```bash
streamlit run app.py
```
Then open the local URL shown in the terminal (usually `http://localhost:8501`).

## 📈 Visualizations Included

1. Age distribution (histogram)
2. Average calories burned by workout type (bar chart)
3. Session duration vs. calories burned (scatter plot)
4. Correlation heatmap of all numeric fitness metrics
5. Fat percentage by gender (boxplot)
6. Workout frequency by experience level (boxplot)
7. BMI distribution across workout types (violin plot)
8. **Interactive dashboard** with live filters (gender, workout type, age, experience)

## 🔍 Key Insights

1. HIIT burns the most calories on average per session compared to other workout types.
2. Session duration and calories burned are strongly positively correlated.
3. Advanced members work out significantly more days per week than beginners.
4. Body fat percentage differs notably between male and female members, consistent with typical physiology.

*(Full auto-generated insights in `outputs/insights.txt`)*

## 🛠️ Technologies Used

- **Python 3** — core language
- **Pandas / NumPy** — data manipulation and generation
- **Matplotlib / Seaborn** — static statistical visualizations
- **Plotly** — interactive charts
- **Streamlit** — web dashboard framework

## 🎓 Possible Extensions

- Connect to a live fitness tracker API (Google Fit / Fitbit) for real-time data
- Add a machine learning model to predict calories burned from other features
- Deploy the Streamlit app publicly (Streamlit Community Cloud)
- Add user authentication to track individual progress over time

---
*Built as a data visualization portfolio/college project.*
