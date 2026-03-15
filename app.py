import streamlit as st
import pandas as pd
import joblib

# =========================================
# LOAD MODEL & ENCODER
# =========================================

model = joblib.load("models/xgb_model.pkl")
encoders = joblib.load("models/encoders.pkl")

# =========================================
# LOAD DATASET
# =========================================

df = pd.read_excel("data/matches.xlsx")

# filter football only
df = df[df["match_main_genre"].astype(str).str.lower() == "football"]

# =========================================
# DROPDOWN DATA FROM DATASET
# =========================================

tournament_list = sorted(df["match_tournament"].dropna().unique())
channel_list = sorted(df["match_channel"].dropna().unique())
team_home_list = sorted(df["team_home"].dropna().unique())
team_away_list = sorted(df["team_away"].dropna().unique())
gender_list = sorted(df["match_gender"].dropna().unique())

# =========================================
# STREAMLIT TITLE
# =========================================

st.title("⚽ Football Match Priority Prediction")

st.write("Machine Learning Prediction using XGBoost")

# =========================================
# USER INPUT
# =========================================

match_hour = st.slider("Match Hour",0,23,20)

match_day = st.selectbox(
    "Match Day",
    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
)

day_map = {
"Monday":0,
"Tuesday":1,
"Wednesday":2,
"Thursday":3,
"Friday":4,
"Saturday":5,
"Sunday":6
}

match_day = day_map[match_day]

match_month = st.slider("Match Month",1,12,1)

match_tournament = st.selectbox(
    "Tournament",
    tournament_list
)

match_channel = st.selectbox(
    "Channel",
    channel_list
)

match_gender = st.selectbox(
    "Match Gender",
    gender_list
)

team_home = st.selectbox(
    "Home Team",
    team_home_list
)

team_away = st.selectbox(
    "Away Team",
    team_away_list
)

# =========================================
# FEATURE ENGINEERING
# =========================================

is_weekend = 1 if match_day in [5,6] else 0
is_prime_time = 1 if 18 <= match_hour <= 23 else 0

# =========================================
# INPUT DATAFRAME
# =========================================

input_data = pd.DataFrame([{

"match_duration":90,

"match_tournament":match_tournament,

"match_premier_status":0,

"match_age_rating":0,

"match_content_type":0,

"match_coverage":0,

"match_genre":0,

"match_main_genre":"football",

"match_channel":match_channel,

"match_gender":match_gender,

"match_organization":0,

"team_home":team_home,

"team_away":team_away,

"match_hour":match_hour,

"match_day":match_day,

"match_month":match_month,

"is_weekend":is_weekend,

"is_prime_time":is_prime_time

}])

# =========================================
# ENCODE DATA
# =========================================

for col, encoder in encoders.items():

    if col in input_data.columns:

        try:

            input_data[col] = encoder.transform(input_data[col].astype(str))

        except:

            input_data[col] = 0


# =========================================
# PREDICTION
# =========================================

if st.button("Predict Priority"):

    prediction = model.predict(input_data)[0]

    label_map = {
        0:"LOW",
        1:"MEDIUM",
        2:"HIGH"
    }

    st.success(f"Match Priority Level : {label_map[prediction]}")