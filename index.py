import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

# ===================================================
# 1. Load Data
# ===================================================
def load_data(path="data.csv"):
    df = pd.read_csv(path)
    df = df.drop(["id", "Unnamed: 32"], axis=1, errors="ignore")
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df

# ===================================================
# 2. Preprocessing
# ===================================================
def preprocess_data(df):
    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

# ===================================================
# 3. Split Data
# ===================================================
def split_data(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# ===================================================
# 4. Train Logistic Regression
# ===================================================
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    return model

# ===================================================
# 5. Evaluate Model
# ===================================================
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.3f}")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    return y_pred, y_prob

# ===================================================
# 6. Threshold Tuning
# ===================================================
def tune_threshold(y_test, y_prob, threshold=0.5):
    y_custom = (y_prob >= threshold).astype(int)
    print(f"\nThreshold: {threshold}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_custom))
    return y_custom

# ===================================================
# Main Pipeline
# ===================================================
def main():
    # Step 1: Load
    df = load_data("data.csv")
    print("Original Data Shape:", df.shape)

    # Step 2: Preprocess
    X, y, scaler = preprocess_data(df)

    # Step 3: Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 4: Train
    model = train_model(X_train, y_train)

    # Step 5: Evaluate
    y_pred, y_prob = evaluate_model(model, X_test, y_test)

    # Step 6: Threshold tuning
    tune_threshold(y_test, y_prob, threshold=0.4)
    tune_threshold(y_test, y_prob, threshold=0.6)


if __name__ == "__main__":
    main()
