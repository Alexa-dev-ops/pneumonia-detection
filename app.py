import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import json
import os

from src.config import MODEL_DIR, OUTPUT_DIR, THRESHOLD, IMG_SIZE
from src.dataset import apply_clahe
from src.explain import make_gradcam_heatmap

st.set_page_config(page_title="Pneumonia Detection AI", layout="wide")

@st.cache_resource
def load_system():
    with open(os.path.join(OUTPUT_DIR, 'results_summary.json'), 'r') as f:
        results = json.load(f)
    weights = results['ensemble_weights']
    
    models = {}
    for arch in weights.keys():
        path = os.path.join(MODEL_DIR, f'{arch}_best.keras')
        if os.path.exists(path):
            models[arch] = tf.keras.models.load_model(path)
    return models, weights

st.title("Pneumonia Detection from Chest X-Rays")
st.write("Upload a chest X-ray image to get a prediction from the 5-model weighted ensemble.")

try:
    models, weights = load_system()
    arch_names = list(weights.keys())
    weight_arr = np.array(list(weights.values()))
    best_model_arch = arch_names[np.argmax(weight_arr)]
    best_model = models[best_model_arch]
except Exception as e:
    st.error("Error loading models. Run train_all.py and evaluate_ensemble.py first.")
    st.stop()

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    img_arr = np.array(image)
    
    # Preprocess
    img_resized = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
    img_clahe = apply_clahe(img_resized)
    input_tensor = np.expand_dims(img_clahe, axis=0)

    # Predict
    preds = []
    for arch in arch_names:
        preds.append(models[arch].predict(input_tensor, verbose=0)[0, 0])
    
    ensemble_prob = np.dot(preds, weight_arr)
    pred_class = "PNEUMONIA" if ensemble_prob >= THRESHOLD else "NORMAL"
    color = "red" if pred_class == "PNEUMONIA" else "green"

    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Original X-Ray', use_container_width=True)
        st.markdown(f"### Prediction: <span style='color:{color}'>{pred_class}</span>", unsafe_allow_html=True)
        st.write(f"**Confidence:** {ensemble_prob:.2%}")

    with col2:
        st.write(f"**Grad-CAM Attention Map** ({best_model_arch})")
        heatmap = make_gradcam_heatmap(input_tensor, best_model)
        if heatmap is not None:
            heatmap_up = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
            heatmap_rgb = cv2.applyColorMap(np.uint8(255 * heatmap_up), cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB) / 255.0
            overlay = img_clahe * 0.6 + heatmap_rgb * 0.4
            st.image(overlay, caption='Focus Areas', use_container_width=True)