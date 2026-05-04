import streamlit as st
from PIL import Image
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

MODEL_PATH = "model/fake_real_detector_weights.npz"
IMG_SIZE = (128, 128)

st.title("Fake vs Real Face Detector")
st.write("Upload a face image and the model will predict whether it is real or fake.")


@st.cache_resource
def load_weights():
    data = np.load(MODEL_PATH)
    return [data[f"arr_{i}"] for i in range(10)]


def relu(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def conv2d_valid(x, kernel, bias):
    patches = sliding_window_view(x, kernel.shape[:3], axis=(0, 1, 2))
    patches = patches[:, :, 0, :, :, :]
    return np.tensordot(patches, kernel, axes=([2, 3, 4], [0, 1, 2])) + bias


def max_pool2d(x):
    h = (x.shape[0] // 2) * 2
    w = (x.shape[1] // 2) * 2
    x = x[:h, :w, :]
    return x.reshape(h // 2, 2, w // 2, 2, x.shape[2]).max(axis=(1, 3))


def predict(image):
    weights = load_weights()
    x = conv2d_valid(image, weights[0], weights[1])
    x = max_pool2d(relu(x))
    x = conv2d_valid(x, weights[2], weights[3])
    x = max_pool2d(relu(x))
    x = conv2d_valid(x, weights[4], weights[5])
    x = max_pool2d(relu(x))
    x = x.reshape(1, -1)
    x = relu(x @ weights[6] + weights[7])
    x = sigmoid(x @ weights[8] + weights[9])
    return float(x[0][0])

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0

    prediction = predict(img_array)

    st.subheader("Prediction")

    if prediction >= 0.5:
        st.error(f"Fake Image Detected")
        st.write(f"Confidence: {prediction * 100:.2f}%")
    else:
        st.success(f"Real Image Detected")
        st.write(f"Confidence: {(1 - prediction) * 100:.2f}%")
