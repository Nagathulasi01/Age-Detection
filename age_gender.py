import sys
print("Python started successfully")

import cv2
import numpy as np

# ----- Load models -----
face_model = "models/res10_300x300_ssd_iter_140000.caffemodel"
face_proto = "models/deploy.prototxt"

age_model = "models/age_net.caffemodel"
age_proto = "models/age_deploy.prototxt"

gender_model = "models/gender_net.caffemodel"
gender_proto = "models/gender_deploy.prototxt"

# Age and gender lists
age_list = ["(0-2)", "(4-6)", "(8-12)", "(15-20)",
            "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
gender_list = ["Male", "Female"]

# Load networks (CORRECT ORDER)
age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)
face_net = cv2.dnn.readNetFromCaffe(face_proto, face_model)

# ----- Start camera -----
cap = cv2.VideoCapture(0)

# Debug
if not cap.isOpened():
    print("Camera failed to open")
    sys.exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera")
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104, 117, 123), swapRB=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            blob2 = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                          (78.426, 87.769, 114.896), swapRB=False)

            gender_net.setInput(blob2)
            gender = gender_list[gender_net.forward()[0].argmax()]

            age_net.setInput(blob2)
            age = age_list[age_net.forward()[0].argmax()]

            label = f"{gender}, {age}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Age & Gender Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
