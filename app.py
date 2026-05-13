"""
Streamlit app for Pneumonia X-ray detection with Grad-CAM explainability.

Usage: upload an X-ray image and the app shows prediction probabilities
and a Grad-CAM overlay.
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import traceback


@st.cache_resource
def load_model():
    candidate = 'best_model (2).keras'
    if not os.path.exists(candidate):
        keras_files = [f for f in os.listdir('.') if f.endswith('.keras')]
        if keras_files:
            candidate = keras_files[0]
    try:
        model = tf.keras.models.load_model(candidate)
        # warmup
        _ = model(tf.zeros((1, 224, 224, 3)), training=False)
        return model, candidate
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        traceback.print_exc()
        return None, candidate


def make_gradcam_heatmap(img_array, model):
    try:
        base_model = model.get_layer('efficientnetb0')
        last_conv = base_model.get_layer('top_activation')

        # submodel to get conv features
        intermediate = tf.keras.Model(inputs=base_model.input, outputs=last_conv.output)

        img_array = tf.cast(img_array, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(img_array)
            conv_outputs = intermediate(img_array)
            conv_outputs = tf.cast(conv_outputs, tf.float32)
            tape.watch(conv_outputs)
            preds = model(img_array, training=False)
            pred_idx = tf.argmax(preds[0])
            class_score = preds[:, pred_idx]

        grads = tape.gradient(class_score, conv_outputs)
        if grads is None:
            return np.zeros((7, 7)), int(pred_idx.numpy()), preds[0].numpy()

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        weighted = conv_outputs[0] * pooled_grads
        heatmap = tf.reduce_sum(weighted, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if tf.math.is_nan(max_val) or max_val == 0:
            heatmap = tf.zeros_like(heatmap)
        else:
            heatmap = heatmap / (max_val + 1e-8)

        return heatmap.numpy(), int(pred_idx.numpy()), preds[0].numpy()
    except Exception as e:
        st.warning(f"Grad-CAM failed: {e}")
        traceback.print_exc()
        preds = model(img_array, training=False)
        return np.zeros((7, 7)), int(tf.argmax(preds[0]).numpy()), preds[0].numpy()


def overlay_heatmap_on_image(orig_img, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    if orig_img.dtype != np.uint8:
        img_uint8 = np.uint8(orig_img * 255)
    else:
        img_uint8 = orig_img
    if len(img_uint8.shape) == 2:
        img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
    overlaid = cv2.addWeighted(img_uint8, 1 - alpha, heatmap_colored, alpha, 0)
    return overlaid


def predict_and_explain(pil_img, model):
    img = pil_img.convert('RGB').resize((224, 224))
    img_np = np.array(img).astype('float32') / 255.0
    inp = np.expand_dims(img_np, axis=0)
    heatmap, pred_idx, preds = make_gradcam_heatmap(inp, model)
    overlay = overlay_heatmap_on_image(np.array(img), heatmap)
    return pred_idx, preds, overlay


def main():
    st.title('🫁 Pneumonia X-ray Detector (Streamlit)')
    st.markdown('Upload a chest X-ray image to get prediction and Grad-CAM overlay.')

    model, model_path = load_model()
    if model is None:
        st.error(f"Model not available (tried: {model_path}). Upload `.keras` file to repo root.")
        return

    uploaded = st.file_uploader('Upload X-ray image', type=['png', 'jpg', 'jpeg'])
    if uploaded is not None:
        pil_img = Image.open(uploaded)
        st.image(pil_img, caption='Input image', use_column_width=True)
        if st.button('Analyze'):
            with st.spinner('Running prediction...'):
                pred_idx, preds, overlay = predict_and_explain(pil_img, model)
                class_names = ['BACTERIA', 'NORMAL', 'VIRUS']
                st.subheader('Predictions')
                for i, name in enumerate(class_names):
                    st.write(f"{name}: {preds[i]:.4f}")
                st.subheader('Grad-CAM overlay')
                st.image(overlay, use_column_width=True)


if __name__ == '__main__':
    main()
