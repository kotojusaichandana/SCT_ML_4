import cv2
import numpy as np
import tensorflow as tf
import os

IMG_SIZE = 64

# Load trained model
model_path = os.path.join("model", "gesture_model.h5")
if not os.path.exists(model_path):
    raise FileNotFoundError("Model file not found. Please run train.py first.")

model = tf.keras.models.load_model(model_path)

# Load gesture labels (from dataset structure)
labels = []
label_path = os.path.join("dataset", "leapGestRecog", "00")
for folder in os.listdir(label_path):
    labels.append(folder)
labels.sort()

# Start webcam
cap = cv2.VideoCapture(0)

print("Press 'Q' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Define Region of Interest (ROI)
    roi = gray[100:300, 100:300]

    # Preprocessing to mimic infrared images
    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)

    # Resize and normalize
    roi = cv2.resize(thresh, (IMG_SIZE, IMG_SIZE))
    roi = roi.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0

    # Predict gesture
    prediction = model.predict(roi, verbose=0)
    gesture = labels[np.argmax(prediction)]

    # Display result
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
    cv2.putText(frame, f"Gesture: {gesture}", (100, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    # Exit on Q key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
