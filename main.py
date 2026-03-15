# ======================================================
# IMPORT LIBRARY
# ======================================================

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier

# ======================================================
# CEK WORKING DIRECTORY
# ======================================================

print("Current directory:", os.getcwd())

# ======================================================
# LOAD DATA
# ======================================================

data_path = Path("data/matches.xlsx")

if not data_path.exists():
    raise FileNotFoundError("Dataset tidak ditemukan di data/matches.xlsx")

df = pd.read_excel(data_path)

print("Shape data awal:", df.shape)

# ======================================================
# FILTER FOOTBALL
# ======================================================

df = df[df["match_main_genre"].astype(str).str.lower() == "football"]

print("Shape setelah filter Football:", df.shape)

# ======================================================
# PILIH KOLOM
# ======================================================

cols = [
    'match_date_start',
    'match_duration',
    'match_tournament',
    'match_premier_status',
    'match_age_rating',
    'match_content_type',
    'match_coverage',
    'match_genre',
    'match_main_genre',
    'match_channel',
    'match_gender',
    'match_organization',
    'team_home',
    'team_away',
    'match_priority_level'
]

df = df[cols].copy()

# ======================================================
# FEATURE ENGINEERING
# ======================================================

df["match_date_start"] = pd.to_datetime(df["match_date_start"])

df["match_hour"] = df["match_date_start"].dt.hour
df["match_day"] = df["match_date_start"].dt.dayofweek
df["match_month"] = df["match_date_start"].dt.month

df["is_weekend"] = df["match_day"].isin([5,6]).astype(int)
df["is_prime_time"] = df["match_hour"].between(18,23).astype(int)

df.drop(columns=["match_date_start"], inplace=True)

# ======================================================
# CLEAN TARGET
# ======================================================

df["match_priority_level"] = (
    df["match_priority_level"]
    .astype(str)
    .str.strip()
    .str.lower()
)

print("\nUnique target values:")
print(df["match_priority_level"].unique())

priority_map = {
    "low":0,
    "medium":1,
    "high":2
}

df["match_priority_level"] = df["match_priority_level"].map(priority_map)

df = df.dropna(subset=["match_priority_level"])

print("\nDistribusi target:")
print(df["match_priority_level"].value_counts())

# ======================================================
# ENCODING CATEGORICAL FEATURE
# ======================================================

encoders = {}

cat_cols = df.select_dtypes(include="object").columns

for col in cat_cols:

    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

    encoders[col] = le

# ======================================================
# SPLIT FEATURE & TARGET
# ======================================================

X = df.drop(columns=["match_priority_level"])
y = df["match_priority_level"]

# ======================================================
# TRAIN TEST SPLIT
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ======================================================
# CLASS WEIGHT
# ======================================================

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))

print("\nClass Weights:")
print(class_weights)

# ======================================================
# TRAIN MODEL
# ======================================================

xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)

xgb_model.fit(X_train, y_train)

# ======================================================
# EVALUASI MODEL
# ======================================================

pred = xgb_model.predict(X_test)

print("\n===== XGBOOST =====")

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# ======================================================
# SIMPAN MODEL
# ======================================================

Path("models").mkdir(exist_ok=True)

joblib.dump(xgb_model, "models/xgb_model.pkl")
joblib.dump(encoders, "models/encoders.pkl")

print("\nModel berhasil disimpan di folder models/")