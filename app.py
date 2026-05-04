import streamlit as st
from PIL import Image
import numpy as np
import sys

MODEL_PATH = "model/fake_real_detector.h5"
IMG_SIZE = (128, 128)

st.title("Fake vs Real Face Detector")
st.write("Upload a face image and the model will predict whether it is real or fake.")

@st.cache_resource
def load_model():
    try:
        import tensorflow as tf
    except ImportError:
        return None

    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

if model is None:
    st.error("TensorFlow is not installed in this Streamlit environment.")
    st.info(
        f"This app is running Python {sys.version.split()[0]}. "
        "TensorFlow is not available for Python 3.14 on Streamlit Cloud. "
        "Delete the app and redeploy it with Python 3.12 in Advanced settings."
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    st.subheader("Prediction")

    if prediction >= 0.5:
        st.error(f"Fake Image Detected")
        st.write(f"Confidence: {prediction * 100:.2f}%")
    else:
        st.success(f"Real Image Detected")
        st.write(f"Confidence: {(1 - prediction) * 100:.2f}%")
