import os
import cv2
import numpy as np
from deepface import DeepFace

# Update paths to your files
prototxt_path = r"D:\VS_Code\DNN_Face_Detector\face_detector\deploy.prototxt"  # since you can't rename it
model_path = r"D:\VS_Code\DNN_Face_Detector\face_detector\res10_300x300_ssd_iter_140000.caffemodel"

# Confirm files exist
assert os.path.isfile(prototxt_path), "Prototxt file not found!"
assert os.path.isfile(model_path), "Model file not found!"

# Load DNN model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame (like your original code)
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # Detect faces using DNN
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Draw face rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Extract the face ROI
            face_roi = frame[y1:y2, x1:x2]

            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
            except Exception as e:
                emotion = "Unknown"

            # Display emotion
            cv2.putText(frame, emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

    cv2.imshow("DNN + Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
