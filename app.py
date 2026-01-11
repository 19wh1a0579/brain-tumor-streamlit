import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet_v2 import preprocess_input

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("ResNet152V2_BrainTumor_Final.h5")

model = load_model()

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
img_size = (224, 224)

st.title("🧠 Brain Tumor Classification with Explainable AI")
st.write("Upload an MRI scan to get prediction, confidence and Grad-CAM explanation.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, 1)
    img_cv = cv2.resize(img_cv, img_size)

    st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), caption="Uploaded MRI", use_column_width=True)

    # Preprocess
    img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img, axis=0)
    img_array = preprocess_input(img_array)

    # Prediction
    preds = model.predict(img_array)
    pred_index = np.argmax(preds)
    pred_class = class_names[pred_index]
    confidence = np.max(preds) * 100

    st.subheader("🔮 Prediction")
    st.write(f"**Class:** {pred_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Grad-CAM
    last_conv_layer = "conv4_block6_out"
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, pred_index]

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
    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_column_width=True)
