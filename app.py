import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import urllib.request
import os
from tensorflow.keras.applications.resnet_v2 import preprocess_input

# -------------------------------
# CONFIG
# -------------------------------
MODEL_URL = "https://huggingface.co/suvidha-reddy/brain-tumor-resnet152v2/resolve/main/ResNet152V2_BrainTumor_Final.h5"

MODEL_PATH = "ResNet152V2_BrainTumor_Final.h5"

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
img_size = (224, 224)

# -------------------------------
# DOWNLOAD + LOAD MODEL
# -------------------------------
@st.cache_resource

def load_model():
    # st.write("🔄 Checking model file...")
    if not os.path.exists(MODEL_PATH):
        st.write("⬇️ Downloading model from Hugging Face...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.write("✅ Download completed")

    # st.write("🧠 Loading model into memory...")
    model = tf.keras.models.load_model(MODEL_PATH)
    # st.write("✅ Model loaded successfully")
    return model


model = load_model()

# -------------------------------
# UI
# -------------------------------
st.title("🧠 Brain Tumor Classification with Explainable AI")
st.write("Upload an MRI scan to get prediction, confidence score, and Grad-CAM heatmap.")
st.warning("⚠️ This tool is for educational purposes only. Not a medical diagnosis.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, 1)
    img_cv = cv2.resize(img_cv, img_size)

    st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), caption="Uploaded MRI", width=300)
    
    # Preprocess
    # Convert to grayscale then back to RGB
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    img_cv = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img_rgb, axis=0)
    img_array = preprocess_input(img_array)

    # Prediction
    preds = model.predict(img_array)
    pred_index = np.argmax(preds)
    pred_class = class_names[pred_index]
    confidence = np.max(preds) * 100

    st.subheader("🔮 Prediction")
    st.write(f"**Class:** {pred_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # -------------------------------
    # GRAD-CAM
    # -------------------------------
    last_conv_layer = "conv4_block6_out"

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[0][:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    heatmap = cv2.resize(heatmap, img_size)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img_cv, 0.75, heatmap, 0.25, 0)

    st.subheader("🧩 Grad-CAM Explanation")
    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), width=350)
