import os
import io
import numpy as np
import streamlit as st
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.cm as cm

# =========================================================
# TC_001: Image Upload Setup & Page Config
# =========================================================
st.set_page_config(
    page_title="Brain Stroke Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI polish
st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .stButton>button { width: 100%; border-radius: 8px; height: 3em; font-weight: bold; }
        h1 { color: #1f77b4; font-family: 'Helvetica', sans-serif; }
        .stAlert { border-radius: 8px; }
        .css-1d391kg { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# PATHS
# =========================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESNET_PATH = os.path.join(MODELS_DIR, "resnet50_best.h5")
EFFNET_PATH = os.path.join(MODELS_DIR, "efficientnetb0_best.h5")
IMG_SIZE = 224

# =========================================================
# HELPER FUNCTIONS (TC_002, TC_004, TC_006)
# =========================================================
# Custom layer to handle 'groups' argument mismatch
class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')  # Remove 'groups' if present
        super().__init__(**kwargs)

@st.cache_resource
def load_models():
    if not os.path.exists(RESNET_PATH) or not os.path.exists(EFFNET_PATH):
        return None, None
    try:
        custom_objects = {'DepthwiseConv2D': FixedDepthwiseConv2D}
        resnet_model = load_model(RESNET_PATH, compile=False, custom_objects=custom_objects)
        eff_model = load_model(EFFNET_PATH, compile=False, custom_objects=custom_objects)
        return resnet_model, eff_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def read_uploaded_image(uploaded_file):
    try:
        img_bytes = uploaded_file.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return pil_img
    except Exception:
        return None

def ct_preprocess_pil(pil_img, img_size=224):
    """TC_002: Apply resizing and normalization."""
    img = np.array(pil_img)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img

def get_stroke_probability(model, img_batch, stroke_index=1):
    """TC_003 & TC_004: Get probability for prediction."""
    pred = model.predict(img_batch, verbose=0)
    pred = np.array(pred)
    if pred.ndim == 2 and pred.shape[1] == 1:
        return float(pred[0][0])
    if pred.ndim == 1 and pred.shape[0] == 1:
        return float(pred[0])
    if pred.ndim == 2 and pred.shape[1] == 2:
        return float(pred[0][stroke_index])
    return 0.0

def infer_stroke_index(model):
    out_shape = model.output_shape
    if isinstance(out_shape, list):
        out_shape = out_shape[0]
    if len(out_shape) == 2 and out_shape[1] == 2:
        return 1
    return None

def find_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub.name
    return None

def get_layer_by_path(model, layer_name):
    if "/" in layer_name:
        parent, child = layer_name.split("/", 1)
        try:
            parent_layer = model.get_layer(parent)
            return parent_layer.get_layer(child)
        except:
            return None
    try:
        return model.get_layer(layer_name)
    except:
        return None

def make_gradcam_heatmap(img_batch, model, last_conv_layer_name, class_index=None):
    """TC_006: Generate Grad-CAM heatmap."""
    if last_conv_layer_name is None:
        return None

    conv_layer = get_layer_by_path(model, last_conv_layer_name)
    if conv_layer is None:
        actual_last_layer = find_last_conv_layer_name(model)
        if actual_last_layer:
             conv_layer = get_layer_by_path(model, actual_last_layer)
        
    if conv_layer is None:
        return None

    grad_model = tf.keras.models.Model([model.inputs], [conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch, training=False)
        if isinstance(predictions, list):
            predictions = predictions[0]

        if predictions.shape[-1] == 1:
            loss = predictions[:, 0]
        else:
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap(base_rgb_224, heatmap_224, alpha=0.40):
    heatmap_uint8 = np.uint8(255 * heatmap_224)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    base = (base_rgb_224 * 255).astype(np.uint8) if base_rgb_224.max() <= 1.0 else base_rgb_224
    overlay = cv2.addWeighted(base, 1 - alpha, heatmap_color, alpha, 0)
    return heatmap_color, overlay

# =========================================================
# MAIN UI
# =========================================================
st.title("🧠 Brain Stroke Detection System")
st.markdown("### AI-Powered CT Scan Analysis")
st.write("Upload a patient's CT scan image to analyze the risk of brain stroke using an ensemble of Deep Learning models.")

# Sidebar - TC_005: Threshold Tuning & Model Weights
st.sidebar.header("⚙️ Settings")
with st.sidebar.expander("Model Parameters (TC_005)", expanded=True):
    threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.6, 0.01, help="Probability threshold to classify as Stroke.")
    w = st.slider("Ensemble Weight (ResNet)", 0.0, 1.0, 0.5, 0.01, help="Weight for ResNet50 in the ensemble.")

st.sidebar.markdown("---")
# Grad-CAM Settings
st.sidebar.header("Explainability")
show_gradcam = st.sidebar.toggle("Show Grad-CAM Heatmaps", value=True)
if show_gradcam:
    gradcam_model_choice = st.sidebar.selectbox("Source Model", ["ResNet50", "EfficientNetB0"])
    smooth_cam = st.sidebar.checkbox("Smooth Heatmap", value=True)

# Load Models
resnet_model, eff_model = load_models()
if not resnet_model or not eff_model:
    st.error("❌ Models not found! Ensure `resnet50_best.h5` and `efficientnetb0_best.h5` are in the `models` folder.")
    st.stop()

# TC_001 & TC_007: Image Upload & Validation
uploaded_file = st.file_uploader("📂 Upload CT Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    pil_img = read_uploaded_image(uploaded_file)
    
    # TC_007: Invalid Upload Handled by checking if read was successful
    if pil_img:
        # Layout: Split Input and Results
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.subheader("1. Patient Scan (TC_001)")
            st.image(pil_img, caption="Original Image", use_container_width=True)
            
            # TC_002: Preprocessing
            img_224 = ct_preprocess_pil(pil_img, IMG_SIZE)
            img_batch = np.expand_dims(img_224, axis=0)
            
            st.markdown("**Preprocessing (TC_002) Status:** ✅ resized to 224x224, normalized.")
            
        with col2:
            st.subheader("2. Analysis Results (TC_003, TC_004)")
            with st.spinner("Running Ensemble Inference..."):
                # TC_003 & TC_004: Prediction & Ensemble
                rs_idx = infer_stroke_index(resnet_model)
                ef_idx = infer_stroke_index(eff_model)
                
                resnet_prob = get_stroke_probability(resnet_model, img_batch, stroke_index=rs_idx if rs_idx else 1)
                effnet_prob = get_stroke_probability(eff_model, img_batch, stroke_index=ef_idx if ef_idx else 1)
                ensemble_prob = (w * resnet_prob) + ((1 - w) * effnet_prob)
                
                # TC_005: Threshold Logic
                final_label = "Stroke" if ensemble_prob >= threshold else "Normal"
                
                # Display Results
                st.metric(label="Stroke Probability (Ensemble)", value=f"{ensemble_prob*100:.2f}%", delta=final_label, delta_color="inverse" if final_label == "Stroke" else "normal")
                
                if final_label == "Stroke":
                    st.error(f"⚠️ **Result: STROKE DETECTED** (Probability ≥ {threshold})")
                else:
                    st.success(f"✅ **Result: NORMAL** (Probability < {threshold})")
                
                with st.expander("Detailed Model Confidence"):
                    st.write(f"**ResNet50:** {resnet_prob*100:.2f}%")
                    st.progress(float(resnet_prob))
                    st.write(f"**EfficientNetB0:** {effnet_prob*100:.2f}%")
                    st.progress(float(effnet_prob))

        # TC_006: Grad-CAM Visualization
        if show_gradcam:
            st.markdown("---")
            st.subheader("3. Explainability (TC_006)")
            st.write("Visualizing regions that contributed most to the prediction.")
            
            model_for_cam = resnet_model if gradcam_model_choice == "ResNet50" else eff_model
            cam_layer = find_last_conv_layer_name(model_for_cam)
            
            if cam_layer:
                heatmap = make_gradcam_heatmap(img_batch, model_for_cam, cam_layer)
                
                if heatmap is not None:
                    heatmap_224 = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
                    if smooth_cam:
                        heatmap_224 = cv2.GaussianBlur(heatmap_224, (11, 11), 0)
                        
                    heat_rgb, overlay = overlay_heatmap(img_224, heatmap_224, alpha=0.40)
                    
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.image(img_224, caption="Preprocessed Input", clamp=True, use_container_width=True)
                    with c2:
                        st.image(heat_rgb, caption=f"Heatmap ({gradcam_model_choice})", use_container_width=True)
                    with c3:
                        st.image(overlay, caption="Grad - CAM", use_container_width=True)
                else:
                    st.warning("Could not generate Grad-CAM heatmap.")
            else:
                st.warning("Could not find suitable layer for Grad-CAM.")

    else:
        # TC_007: Fail state
        st.error("❌ **Error (TC_007): Unrecognized File Format.** Please upload a valid JPG or PNG image.")
else:
    # Empty state
    st.info("👆 Awaiting file upload to start analysis.")
