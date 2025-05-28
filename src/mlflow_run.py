import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
import xgboost as xgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

def wrist_relative_scaling(landmarks):
    landmarks = np.array(landmarks).copy()
    
    if landmarks.ndim == 2: 
        landmarks = landmarks.reshape(-1, 21, 2)
    elif landmarks.ndim == 3:  
        pass
    else:
        raise ValueError("Input must be 2D (N,42) or 3D (N,21,2)")
    
    wrist = landmarks[:, 0, :] 
    middle_tip = landmarks[:, 12, :]
    
    # Compute scale
    scale = np.linalg.norm(middle_tip - wrist, axis=1, keepdims=True)
    scale[scale == 0] = 1  # Avoid division by zero
    
    # Normalize landmarks: (landmarks - wrist) / scale
    landmarks = (landmarks - wrist[:, np.newaxis, :]) / scale[:, np.newaxis, :]
    
    # Flatten
    return landmarks.reshape(landmarks.shape[0], -1)

#load data
df= pd.read_csv(r'C:\Users\Legion\VS_Notebooks\Real-Time_Gesture_Game_Control\data\hand_landmarks_data.csv')

#create copy of df
df_copy = df.copy()

#drop rows with values not in allowed list
allowed = ['one', 'two_up', 'three', 'four']
last_col = df_copy.columns[-1]
df_copy = df_copy[df_copy[last_col].isin(allowed)]

#drop columns
columns_to_drop = df_copy.columns[2::3]
df_copy = df_copy.drop(columns=columns_to_drop)

#shuffle dataset
df_shuffled = df_copy.sample(frac=1, random_state=42).reset_index(drop=True)
X = df_shuffled.drop(df_shuffled.columns[-1], axis=1)
y = df_shuffled[df_shuffled.columns[-1]]

#split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify= y)

# Apply scaling
X_scaled = wrist_relative_scaling(X_train)
X_test_scaled = wrist_relative_scaling(X_test)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# === Logistic Regression Logging ===
print("Running Logistic Regression...")

# Load model (already trained and saved)
logreg = joblib.load(r"src/logreg_model.pkl")

X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_train.columns)

# Predict (predictions are strings, matching training labels)
preds_logreg = logreg.predict(X_test_scaled_df)

# Convert string predictions to encoded ints
preds_logreg_encoded = label_encoder.transform(preds_logreg)

# Evaluate with encoded labels
acc_logreg = accuracy_score(y_test_encoded, preds_logreg_encoded)
f1_logreg = f1_score(y_test_encoded, preds_logreg_encoded, average='macro')

# MLflow logging
with mlflow.start_run(run_name="Logistic_Regression_Multiclass"):
    mlflow.log_params(logreg.get_params())
    mlflow.log_metric("accuracy", acc_logreg)
    mlflow.log_metric("f1_score", f1_logreg)

    mlflow.sklearn.log_model(logreg, "model")
    mlflow.log_artifact(r"src/logreg_model.pkl")

    print(f"Logged Logistic Regression: Accuracy={acc_logreg:.4f}, F1 Score={f1_logreg:.4f}")

# === XGBoost Logging ===
print("\nRunning XGBoost...")

# Load model
xgb_model = xgb.Booster()
xgb_model.load_model(r"src/xgboost_model.model")

# Predict
dtest = xgb.DMatrix(X_test_scaled)
preds_xgb_prob = xgb_model.predict(dtest)
preds_xgb_labels = np.argmax(preds_xgb_prob, axis=1)

# Evaluate
acc_xgb = accuracy_score(y_test_encoded, preds_xgb_labels)
f1_xgb = f1_score(y_test_encoded, preds_xgb_labels, average='macro')

# Save a copy
xgb_model.save_model("xgboost_model_logged.model")

# MLflow logging
with mlflow.start_run(run_name="XGBoost_Multiclass"):
    mlflow.log_metric("accuracy", acc_xgb)
    mlflow.log_metric("f1_score", f1_xgb)

    mlflow.xgboost.log_model(xgb_model, artifact_path="model")
    mlflow.log_artifact("xgboost_model_logged.model")

    print(f"Logged XGBoost: Accuracy={acc_xgb:.4f}, F1 Score={f1_xgb:.4f}")
