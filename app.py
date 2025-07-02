import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
from PIL import Image
import os
import datetime

# ------------------------
# Function to check if it's a valid skin-like image
def is_skin_image(img_array):
    """
    Basic rule-based skin detector based on average color.
    Blocks animals, cartoons, or random non-skin images.
    """
    avg_color = np.mean(img_array, axis=(0, 1))
    if 30 < avg_color[0] < 240 and 30 < avg_color[1] < 200 and 30 < avg_color[2] < 200:
        return True
    return False

# ------------------------
# Function to log detected case to text file
def log_case(predicted_label, confidence):
    with open("detected_cases.txt", "a") as f:
        f.write(f"{datetime.datetime.now()} - {predicted_label} - Confidence: {confidence:.2f}\n")

# ------------------------
# Download model if not already present
MODEL_PATH = "skin_cancer_model.h5"
if not os.path.exists(MODEL_PATH):
    file_id = "12feZzigKSdMMo1ienehOu-SUYpyNcgS2"
    gdown.download(f"https://drive.google.com/uc?id={file_id}", MODEL_PATH, quiet=False)

# Load the model
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

# ------------------------
# Streamlit UI
st.title("🧠 Skin Cancer Detector")
st.write("Upload a skin image and we'll predict the cancer type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', use_container_width=True)

    # Resize and preprocess
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0

    # Check if image looks like real skin
    if not is_skin_image(img_array * 255):
        st.error("🚫 This doesn't appear to be a valid skin image. Please upload a proper skin lesion photo.")
    else:
        # Predict
        img_input = np.expand_dims(img_array, axis=0)
        raw_prediction = model.predict(img_input)
        prediction = raw_prediction  # Assuming model ends with softmax
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        # ✅ Log the result
        log_case(label_map[predicted_class], confidence)

        # Show result
        st.markdown(f"### 🔍 Prediction: **{label_map[predicted_class]}**")
        st.markdown(f"Confidence: `{confidence:.2f}`")

        if confidence < 0.70:
            st.error("⚠️ Uncertain result. Try uploading a clearer skin lesion image taken under good lighting.")

# ------------------------
# Footer Disclaimer
st.markdown("""
---
📝 **Disclaimer**: This tool is for educational and research purposes only.  
It does not replace professional medical advice or diagnosis.  
Always consult a dermatologist for clinical evaluation.
""")
