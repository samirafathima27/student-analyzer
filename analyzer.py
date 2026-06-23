import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def load_and_clean(file):
    """Load and clean the dataset"""
    df = pd.read_csv(file)

    # Rename columns for easier use
    df.columns = [
        "gender", "race", "parent_education",
        "lunch", "test_prep", "math", "reading", "writing"
    ]

    # Add total and average score
    df["total"] = df["math"] + df["reading"] + df["writing"]
    df["average"] = (df["total"] / 3).round(2)

    # Add pass/fail column — pass if average >= 50
    df["result"] = df["average"].apply(
        lambda x: "Pass" if x >= 50 else "Fail"
    )

    # Add performance category
    df["performance"] = pd.cut(
        df["average"],
        bins=[0, 50, 60, 75, 90, 100],
        labels=["Fail", "Below Average", "Average", "Good", "Excellent"]
    )

    return df


def get_overview(df):
    """Return basic overview statistics"""
    overview = {
        "total_students": len(df),
        "pass_count": len(df[df["result"] == "Pass"]),
        "fail_count": len(df[df["result"] == "Fail"]),
        "avg_math": round(df["math"].mean(), 2),
        "avg_reading": round(df["reading"].mean(), 2),
        "avg_writing": round(df["writing"].mean(), 2),
        "avg_total": round(df["average"].mean(), 2),
        "top_student": df.loc[df["average"].idxmax(), "average"],
        "lowest_student": df.loc[df["average"].idxmin(), "average"],
    }
    return overview


def get_gender_analysis(df):
    """Average scores by gender"""
    return df.groupby("gender")[["math", "reading", "writing", "average"]].mean().round(2)


def get_test_prep_analysis(df):
    """Impact of test preparation on scores"""
    return df.groupby("test_prep")[["math", "reading", "writing", "average"]].mean().round(2)


def get_parent_education_analysis(df):
    """Average scores by parental education"""
    return df.groupby("parent_education")["average"].mean().round(2).sort_values(ascending=False)


def get_performance_distribution(df):
    """Count of students in each performance category"""
    return df["performance"].value_counts()


def train_model(df):
    """
    Train a Random Forest model to predict pass/fail
    Returns model and accuracy score
    """
    # Make a copy to avoid changing original
    model_df = df.copy()

    # Encode categorical columns to numbers
    le = LabelEncoder()
    model_df["gender_enc"] = le.fit_transform(model_df["gender"])
    model_df["lunch_enc"] = le.fit_transform(model_df["lunch"])
    model_df["test_prep_enc"] = le.fit_transform(model_df["test_prep"])
    model_df["result_enc"] = le.fit_transform(model_df["result"])

    # Features and target
    X = model_df[["gender_enc", "lunch_enc", "test_prep_enc", "math", "reading", "writing"]]
    y = model_df["result_enc"]

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Check accuracy
    y_pred = model.predict(X_test)
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)

    return model, accuracy, le


def predict_student(model, le, gender, lunch, test_prep, math, reading, writing):
    """Predict if a single student will pass or fail"""
    gender_enc = 1 if gender == "female" else 0
    lunch_enc = 1 if lunch == "standard" else 0
    test_prep_enc = 1 if test_prep == "completed" else 0

    features = [[gender_enc, lunch_enc, test_prep_enc, math, reading, writing]]
    prediction = model.predict(features)[0]

    return "Pass ✅" if prediction == 1 else "Fail ❌"