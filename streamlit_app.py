import streamlit as st
import pandas as pd
import joblib

# load model
model = joblib.load("models/xgb_model.pkl")

st.set_page_config(page_title="Football Match Priority", layout="centered")

st.title("⚽ Football Match Priority Prediction")

st.markdown("Predict how important a football match is based on match schedule.")

st.divider()

# TEAM INPUT
col1, col2 = st.columns(2)

with col1:
    team_home = st.text_input("🏠 Home Team")

with col2:
    team_away = st.text_input("✈️ Away Team")

st.divider()

# MATCH INFO
match_duration = st.number_input("Match Duration (minutes)", value=90)

match_hour = st.slider("Kick Off Hour", 0, 23)

match_day = st.slider("Day of Week (0=Monday)", 0, 6)

match_month = st.slider("Month", 1, 12)

col3, col4 = st.columns(2)

with col3:
    is_weekend = st.selectbox("Weekend Match?", [0,1])

with col4:
    is_prime_time = st.selectbox("Prime Time Match?", [0,1])


st.divider()

# PREDICT
if st.button("🔎 Predict Match Priority"):

    data = pd.DataFrame({
        "match_duration":[match_duration],
        "match_hour":[match_hour],
        "match_day":[match_day],
        "match_month":[match_month],
        "is_weekend":[is_weekend],
        "is_prime_time":[is_prime_time]
    })

    pred = model.predict(data)[0]

    label_map = {
        0:"Low",
        1:"Medium",
        2:"High"
    }

    st.success(f"Match Priority Prediction: **{label_map[pred]}**")