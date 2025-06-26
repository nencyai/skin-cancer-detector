import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import gdown
from PIL import Image
import os

# Download model if not present
MODEL_PATH = "skin_cancer_model.h5"
if not os.path.exists(MODEL_PATH):
    file_id = "12feZzigKSdMMo1ienehOu-SUYpyNcgS2"  # Your Drive file ID
    gdown.download(f"https://drive.google.com/uc?id={file_id}", MODEL_PATH, quiet=False)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Label map (adjust if needed)
label_map = {
    0: 'akiec',
    1: 'bcc',
    2: 'bkl',
    3: 'df',
    4: 'mel',
    5: 'nv',
    6: 'vasc'
}

# Streamlit UI
st.title("🧠 Skin Cancer Detector")
st.write("Upload a skin image and we'll predict the cancer type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img_array = np.array(img)
    img_resized = cv2.resize(img_array, (224, 224)) / 255.0
    img_input = np.expand_dims(img_resized, axis=0)

    # Predict
    prediction = model.predict(img_input)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.markdown(f"### 🔍 Prediction: **{label_map[predicted_class]}**")
    st.markdown(f"Confidence: `{confidence:.2f}`")
