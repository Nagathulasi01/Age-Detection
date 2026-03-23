import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Configure Streamlit page
st.set_page_config(page_title="Age & Gender Detection", page_icon="👤", layout="centered")

st.title("👤 Age and Gender Detection")
st.write("Real-time Age and Gender detection using OpenCV and Streamlit WebRTC.")

@st.cache_resource
def load_models():
    """Load the models. Stored in cache to avoid reloading on each frame."""
    face_model = "models/res10_300x300_ssd_iter_140000.caffemodel"
    face_proto = "models/deploy.prototxt"
    age_model = "models/age_net.caffemodel"
    age_proto = "models/age_deploy.prototxt"
    gender_model = "models/gender_net.caffemodel"
    gender_proto = "models/gender_deploy.prototxt"

    face_net = cv2.dnn.readNetFromCaffe(face_proto, face_model)
    age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
    gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)

    return face_net, age_net, gender_net

face_net, age_net, gender_net = load_models()

age_list = ["(0-2)", "(4-6)", "(8-12)", "(15-20)",
            "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
gender_list = ["Male", "Female"]

class AgeGenderTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300),
                                     (104, 117, 123), swapRB=False)
        face_net.setInput(blob)
        detections = face_net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                
                # Bounds check
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                face = img[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                blob2 = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                              (78.426, 87.769, 114.896), swapRB=False)

                gender_net.setInput(blob2)
                gender = gender_list[gender_net.forward()[0].argmax()]

                age_net.setInput(blob2)
                age = age_list[age_net.forward()[0].argmax()]

                label = f"{gender}, {age}"
                
                # Draw Box and Label
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return img

st.markdown("### Live Webcam Feed")
st.write("Click 'Start' below to allow camera access and run the detection.")

webrtc_streamer(
    key="age-gender-detection",
    video_transformer_factory=AgeGenderTransformer,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

st.markdown("---")
st.markdown("""
**Note for deployment**: If you are hosting on Streamlit Community Cloud and the camera isn't working, 
ensure that you are accessing the app via `HTTPS`. Modern browsers restrict camera access over HTTP.
""")
