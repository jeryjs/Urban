# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import joblib
import os

DATA_FILE = "simulated_farm_data.csv"
IRR_MODEL_FILE = "irrigation_model.joblib"
GROWTH_MODEL_FILE = "growth_model.joblib"


def load_data(filename=DATA_FILE):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found. Run generate_data.py first.")
    return pd.read_csv(filename)


def preprocess(df):
    df2 = df.copy()

    # One-hot encode crop and stage
    df2 = pd.get_dummies(df2, columns=["crop", "stage"], drop_first=True)

    feature_cols = ["moisture", "nutrients", "light_hours", "temperature"]
    feature_cols += [c for c in df2.columns if c.startswith("crop_") or c.startswith("stage_")]

    X = df2[feature_cols]

    return X, df2, feature_cols


def train_irrigation_model(X, y):
    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(X, y)
    return model


def train_growth_model(X, y):
    model = RandomForestRegressor(n_estimators=140, random_state=42)
    model.fit(X, y)
    return model


def main():
    print("Loading dataset...")
    df = load_data()

    print("Preprocessing...")
    X, df2, feature_cols = preprocess(df)

    # --------------------------
    # Irrigation Classification Model
    # --------------------------
    y_irrigation = df2["irrigation_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_irrigation, test_size=0.2, random_state=42, stratify=y_irrigation
    )

    print("\nTraining Irrigation Model...")
    irrigation_model = train_irrigation_model(X_train, y_train)
    preds_irrigation = irrigation_model.predict(X_test)

    print("\n=== Irrigation classification report ===")
    print(classification_report(y_test, preds_irrigation))

    # --------------------------
    # Growth Regression Model
    # --------------------------
    y_growth = df2["growth_index"]

    X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
        X, y_growth, test_size=0.2, random_state=42
    )

    print("\nTraining Growth Prediction Model...")
    growth_model = train_growth_model(X_train_g, y_train_g)
    preds_growth = growth_model.predict(X_test_g)

    # FIXED RMSE (manual calculation)
    mse = mean_squared_error(y_test_g, preds_growth)
    rmse = mse ** 0.5  # manual RMSE calculation

    print("\n=== Growth regressor metrics ===")
    print("RMSE =", rmse)
    print("R2 Score =", r2_score(y_test_g, preds_growth))

    # --------------------------
    # Save Models & Features
    # --------------------------
    joblib.dump(irrigation_model, IRR_MODEL_FILE)
    joblib.dump(growth_model, GROWTH_MODEL_FILE)
    joblib.dump(feature_cols, "model_feature_columns.joblib")

    print("\nSaved model files:")
    print(" -", IRR_MODEL_FILE)
    print(" -", GROWTH_MODEL_FILE)
    print(" - model_feature_columns.joblib")


if __name__ == "__main__":
    main()
