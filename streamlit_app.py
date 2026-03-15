import streamlit as st
import pandas as pd
import joblib

# load model
model = joblib.load("models/xgb_model.pkl")

st.title("Football Match Priority Prediction")

st.write("Predict match priority level")

match_duration = st.number_input("Match Duration")

match_hour = st.slider("Match Hour",0,23)

match_day = st.slider("Match Day (0=Monday)",0,6)

match_month = st.slider("Match Month",1,12)

is_weekend = st.selectbox("Is Weekend",[0,1])

is_prime_time = st.selectbox("Prime Time",[0,1])


if st.button("Predict"):

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

    st.success(f"Prediction: {label_map[pred]}")