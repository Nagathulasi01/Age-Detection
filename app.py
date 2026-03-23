import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Configure Streamlit page
st.set_page_config(page_title="Age & Gender Detection", page_icon="👤", layout="centered")

st.title("👤 Age and Gender Detection")
st.write("Upload an image or take a picture using your webcam to detect age and gender.")

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

def process_image(image_bytes):
    # Convert uploaded image to OpenCV format
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300),
                                 (104, 117, 123), swapRB=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    faces_found = False
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            faces_found = True
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

    return img, faces_found

st.markdown("---")

# Option to either Upload an Image OR use the Camera
option = st.radio("Select an Input Method:", ("Upload Image", "Take a Photo with Camera"))

image_file = None

if option == "Upload Image":
    image_file = st.file_uploader("Upload a photo (JPEG/PNG)", type=['jpg', 'jpeg', 'png'])
elif option == "Take a Photo with Camera":
    image_file = st.camera_input("Take a picture")

if image_file is not None:
    st.markdown("### Detection Results")
    with st.spinner("Analyzing image..."):
        processed_img, face_detected = process_image(image_file)
        
        # Convert BGR (OpenCV) back to RGB to display in Streamlit
        processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        
        st.image(processed_img_rgb, channels="RGB", use_container_width=True)
        
        if not face_detected:
            st.warning("No faces were detected with high confidence in this image.")
