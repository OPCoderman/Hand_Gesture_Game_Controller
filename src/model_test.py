import numpy as np
import pandas as pd
import cv2 as cv
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles
import xgboost as xgb
from collections import deque

# Load XGBoost model
model = xgb.Booster()
model.load_model(r"C:\Users\Legion\VS_Notebooks\Real-Time_Gesture_Game_Control\src\xgboost_model_logged.model")

# Define hand sign classes
hand_sign_classes = ['left', 'up', 'down', 'right']

# Feature names - now only using x and y coordinates (42 features instead of 63)
feature_names = [f"f{i}" for i in range(42)]  # 21 landmarks * 2 (x, y)

# Initialize Mediapipe Hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# Set camera resolution
width, height = 1280, 720
cam = cv.VideoCapture(0)
cam.set(3, width)
cam.set(4, height)

# Stabilization window for smoother predictions
prediction_window = deque(maxlen=5)

# Video capture loop
while cam.isOpened():
    success, img = cam.read()
    if not success:
        print("Camera Frame not available")
        continue

    # Convert image to RGB for Mediapipe
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    hands_detected = hands.process(img_rgb)

    # Convert back to BGR for OpenCV display
    img_rgb = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)

    # Default hand sign message when no hands are detected
    hand_sign = "No hand detected"

    # If hands are detected
    if hands_detected.multi_hand_landmarks:
        for hand_landmarks in hands_detected.multi_hand_landmarks:
            # Draw hand landmarks
            drawing.draw_landmarks(
                img_rgb,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style()
            )

            # Extract only x and y landmarks (ignoring z)
            landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])

            # Normalize landmarks: Recenter to wrist (landmark[0]) and scale by middle finger tip (landmark[12])
            wrist = landmarks[0]
            middle_tip = landmarks[12]
            scale = np.linalg.norm(middle_tip - wrist) if np.linalg.norm(middle_tip - wrist) != 0 else 1
            landmarks = (landmarks - wrist) / scale  # Normalize x, y

            # Flatten landmarks for model input
            landmarks = landmarks.flatten().reshape(1, -1)

            # Convert landmarks to DMatrix
            dmatrix = xgb.DMatrix(landmarks)

            # Predict hand sign
            prediction = model.predict(dmatrix)

            # Debug prints
            print("Raw prediction:", prediction)

            # Get the predicted class (max probability or score)
            predicted_class = int(np.argmax(prediction))

            # Stabilize prediction with mode of recent predictions
            prediction_window.append(predicted_class)
            stabilized_class = max(set(prediction_window), key=prediction_window.count)

            # Get the hand sign label
            hand_sign = hand_sign_classes[stabilized_class]

    # Show video feed with flipped image
    flipped_img = cv.flip(img_rgb, 1)

    # Display hand sign prediction
    cv.putText(
        flipped_img,
        f"Hand Sign: {hand_sign}",
        (width - 800, 50),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
        cv.LINE_AA
    )

    cv.putText(
        flipped_img,
        "Press Q to quit",
        (50, height - 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
        cv.LINE_AA
    )

    # Show the video window
    cv.imshow("Hand Sign Recognition", flipped_img)

    # Exit on 'q' key
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

# Release resources
cam.release()
cv.destroyAllWindows()
