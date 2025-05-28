import numpy as np
import xgboost as xgb

model = xgb.Booster()
model.load_model(r"src/model.model")

hand_sign_classes = ['left', 'up', 'down', 'right']

def predict_hand_sign(landmarks):
    landmarks = np.array(landmarks)
    landmarks_xy = landmarks[:, :2]
    wrist = landmarks_xy[0]
    middle_tip = landmarks_xy[12]
    scale = np.linalg.norm(middle_tip - wrist) or 1
    normalized = (landmarks_xy - wrist) / scale
    flattened = normalized.flatten().reshape(1, -1)
    dmatrix = xgb.DMatrix(flattened)
    prediction = model.predict(dmatrix)
    predicted_class = int(np.argmax(prediction))
    return hand_sign_classes[predicted_class]
