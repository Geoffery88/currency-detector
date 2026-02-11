import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import matplotlib.cm as cm
import tempfile
import os

# --- CONFIGURATION ---
IMG_SIZE = (224, 224)
# ResNet50 specific layer for Grad-CAM
LAST_CONV_LAYER = 'conv5_block3_out' 
CLASS_NAMES = ['Fake', 'Real']

st.set_page_config(page_title="Dynamic Currency Detector", layout="wide")

# --- GRAD-CAM FUNCTIONS ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(heatmap, image, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((image.shape[1], image.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + image
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img

# --- APP UI ---
st.title("💸 Indian Currency Detector")
st.sidebar.header("Step 1: Upload Model")

# Sidebar for Model Upload
uploaded_model = st.sidebar.file_uploader("Upload your trained .h5 model", type=["h5"])

# Load model logic
model = None
if uploaded_model is not None:
    try:
        # Keras needs a physical file path to load, so we save the upload to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_file:
            tmp_file.write(uploaded_model.getvalue())
            tmp_path = tmp_file.name
        
        model = tf.keras.models.load_model(tmp_path)
        st.sidebar.success("✅ Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load model: {e}")

st.divider()

if model is None:
    st.info(" Please upload your `currency_resnet_local.h5` model in the sidebar to begin.")
else:
    st.subheader("Step 2: Upload Currency Note Image")
    uploaded_image = st.file_uploader("Choose a note image...", type=["jpg", "png", "jpeg"])

    if uploaded_image:
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_image)
        
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)

        # Preprocessing
        img_array = np.array(ImageOps.fit(image, IMG_SIZE, Image.Resampling.LANCZOS))
        if img_array.shape[-1] == 4: img_array = img_array[..., :3]
        img_batch = np.expand_dims(img_array, axis=0).astype('float32') / 255.0

        # Prediction
        preds = model.predict(img_batch)
        class_idx = np.argmax(preds)
        confidence = np.max(preds)
        label = CLASS_NAMES[class_idx]

        # Grad-CAM Visualization
        try:
            heatmap = make_gradcam_heatmap(img_batch, model, LAST_CONV_LAYER)
            grad_cam = overlay_heatmap(heatmap, np.array(image))
            with col2:
                st.image(grad_cam, caption="AI Feature Analysis (Heatmap)", use_column_width=True)

            st.write("---")
            if label == "Real":
                st.success(f"### Prediction: REAL ({confidence:.1%})")
            else:
                st.error(f"### Prediction: FAKE ({confidence:.1%})")
        except Exception as e:
            st.error(f"Visualization error: {e}")