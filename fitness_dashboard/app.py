"""
app.py
------
Interactive Fitness Data Visualization Dashboard
Run with: streamlit run app.py
"""
# //streamlit run app.py
import joblib
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Fitness Analytics Dashboard", layout="wide", page_icon="💪")

DATA_PATH = "data/gym_members_exercise_tracking.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()

# ---------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------
st.sidebar.header("🔍 Filters")

gender_filter = st.sidebar.multiselect(
    "Gender", options=df["Gender"].unique(), default=list(df["Gender"].unique())
)
workout_filter = st.sidebar.multiselect(
    "Workout Type", options=df["Workout_Type"].unique(), default=list(df["Workout_Type"].unique())
)
age_range = st.sidebar.slider(
    "Age Range", int(df["Age"].min()), int(df["Age"].max()),
    (int(df["Age"].min()), int(df["Age"].max()))
)
exp_filter = st.sidebar.multiselect(
    "Experience Level", options=sorted(df["Experience_Level"].unique()),
    default=sorted(df["Experience_Level"].unique())
)

filtered_df = df[
    (df["Gender"].isin(gender_filter))
    & (df["Workout_Type"].isin(workout_filter))
    & (df["Age"].between(age_range[0], age_range[1]))
    & (df["Experience_Level"].isin(exp_filter))
]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(filtered_df)}** of {len(df)} members")

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.title("💪 Fitness Analytics Dashboard")
st.caption("Data Visualization Project — Python, Pandas, Plotly, Streamlit")

# ---------------------------------------------------------------
# KPI Row
# ---------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Members", len(filtered_df))
col2.metric("Avg Calories Burned", f"{filtered_df['Calories_Burned'].mean():.0f} kcal")
col3.metric("Avg Session Duration", f"{filtered_df['Session_Duration (hours)'].mean():.2f} hrs")
col4.metric("Avg BMI", f"{filtered_df['BMI'].mean():.1f}")

st.markdown("---")

# ---------------------------------------------------------------
# Row 1: Calories by workout type + Age distribution
# ---------------------------------------------------------------
c1, c2 = st.columns(2)

with c1:
    st.subheader("Avg Calories Burned by Workout Type")
    cal_by_type = filtered_df.groupby("Workout_Type", as_index=False)["Calories_Burned"].mean()
    fig1 = px.bar(
        cal_by_type, x="Workout_Type", y="Calories_Burned", color="Workout_Type",
        text_auto=".0f", color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig1.update_layout(showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    st.subheader("Age Distribution")
    fig2 = px.histogram(filtered_df, x="Age", nbins=20, color_discrete_sequence=["#2E86AB"])
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------
# Row 2: Scatter + Correlation heatmap
# ---------------------------------------------------------------
c3, c4 = st.columns(2)

with c3:
    st.subheader("Session Duration vs Calories Burned")
    fig3 = px.scatter(
        filtered_df, x="Session_Duration (hours)", y="Calories_Burned",
        color="Workout_Type", size="BMI", hover_data=["Age", "Gender"],
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    st.subheader("Correlation Heatmap")
    numeric_df = filtered_df.select_dtypes(include="number")
    corr = numeric_df.corr()
    fig4 = go.Figure(
        data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale="RdBu", zmid=0, text=corr.round(2).values,
            texttemplate="%{text}", textfont={"size": 8}
        )
    )
    fig4.update_layout(height=450)
    st.plotly_chart(fig4, use_container_width=True)

# ---------------------------------------------------------------
# Row 3: Fat % by gender + Frequency by experience
# ---------------------------------------------------------------
c5, c6 = st.columns(2)

with c5:
    st.subheader("Fat Percentage by Gender")
    fig5 = px.box(filtered_df, x="Gender", y="Fat_Percentage", color="Gender",
                  color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig5, use_container_width=True)

with c6:
    st.subheader("Workout Frequency by Experience Level")
    fig6 = px.box(
        filtered_df, x="Experience_Level", y="Workout_Frequency (days/week)",
        color="Experience_Level", color_discrete_sequence=px.colors.qualitative.Prism
    )
    st.plotly_chart(fig6, use_container_width=True)

# ---------------------------------------------------------------
# Raw data
# ---------------------------------------------------------------
st.markdown("---")
with st.expander("📄 View Raw Data"):
    st.dataframe(filtered_df, use_container_width=True)

    # ---------------------------------------------------------------
# Predict Calories Burned (ML section)
# ---------------------------------------------------------------
st.markdown("---")
st.header("🤖 Predict Your Calories Burned")
st.caption("Enter your details below and let the trained ML model estimate calories burned per session.")

try:
    model = joblib.load("src/calories_model.pkl")
    encoder = joblib.load("src/workout_encoder.pkl")

    p1, p2, p3 = st.columns(3)
    with p1:
        p_age = st.number_input("Age", 15, 80, 25)
        p_weight = st.number_input("Weight (kg)", 40.0, 150.0, 70.0)
    with p2:
        p_height = st.number_input("Height (m)", 1.4, 2.2, 1.70)
        p_duration = st.number_input("Session Duration (hours)", 0.25, 3.0, 1.0)
    with p3:
        p_bpm = st.number_input("Avg BPM during workout", 90, 190, 130)
        p_workout = st.selectbox("Workout Type", encoder.classes_)
        p_exp = st.selectbox("Experience Level", [1, 2, 3])

    if st.button("Predict Calories Burned"):
        workout_encoded = encoder.transform([p_workout])[0]
        input_data = np.array([[p_age, p_weight, p_height, p_duration, p_bpm, workout_encoded, p_exp]])
        prediction = model.predict(input_data)[0]
        st.success(f"🔥 Estimated Calories Burned: **{prediction:.0f} kcal**")

except FileNotFoundError:
    st.warning("Model file not found. Run `python src/ml_model.py` first to train the model.")

st.caption("Built with Streamlit + Plotly + Pandas | Fitness Data Visualization Project")
