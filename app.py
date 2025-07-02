import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
from PIL import Image
import os

# Download model if not already present
MODEL_PATH = "skin_cancer_model.h5"
if not os.path.exists(MODEL_PATH):
    file_id = "12feZzigKSdMMo1ienehOu-SUYpyNcgS2"
    gdown.download(f"https://drive.google.com/uc?id={file_id}", MODEL_PATH, quiet=False)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Label map
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
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', use_container_width=True)

    # Resize using PIL
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0

    # 🔍 Skin image check BEFORE prediction
    if not is_skin_image(img_array * 255):  # multiply to get back 0–255 range
        st.error("🚫 This doesn't look like a skin image. Please upload a valid skin lesion image.")
    else:
        img_input = np.expand_dims(img_array, axis=0)
        raw_prediction = model.predict(img_input)
        prediction = tf.nn.softmax(raw_prediction)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        # Show result
        st.markdown(f"### 🔍 Prediction: **{label_map[predicted_class]}**")
        st.markdown(f"Confidence: `{confidence:.2f}`")

        if confidence < 0.70:
            st.warning("⚠️ This result may be uncertain. Please consult a dermatologist.")
