import os
import cv2
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess



# -----------------------------
# Basic image load
# -----------------------------
def load_image_rgb(pil_img, target_size=(224, 224)):
    """Convert uploaded PIL image to RGB numpy array resized."""
    img = pil_img.convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img)
    return arr


# -----------------------------
# Preprocess for each backbone
# -----------------------------
def preprocess_for_resnet(pil_img, img_size=224):
    arr = load_image_rgb(pil_img, (img_size, img_size)).astype(np.float32)
    x = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    x = resnet_preprocess(x)
    return x


def preprocess_for_effnet(pil_img, img_size=224):
    arr = load_image_rgb(pil_img, (img_size, img_size)).astype(np.float32)
    x = np.expand_dims(arr, axis=0)
    x = eff_preprocess(x)
    return x


# -----------------------------
# Predict helpers
# -----------------------------
def predict_binary_sigmoid(model, x):
    """
    Returns probability for class 'Stroke' assuming output sigmoid.
    Handles (1,1) or (1,) outputs.
    """
    p = model.predict(x, verbose=0)
    p = np.array(p).reshape(-1)
    return float(p[0])


def ensemble_probability(p_resnet, p_eff, w=0.5):
    """Weighted ensemble: w*resnet + (1-w)*effnet"""
    return w * p_resnet + (1 - w) * p_eff


def classify_from_prob(p, threshold=0.6):
    label = "Stroke" if p >= threshold else "Normal"
    confidence = p if label == "Stroke" else (1 - p)
    return label, confidence


# -----------------------------
# Grad-CAM
# -----------------------------
def make_gradcam_heatmap(input_tensor, model, last_conv_layer_name, pred_index=None):
    """
    Standard Grad-CAM for a Keras model.
    Works with sigmoid output too.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor)
        if pred_index is None:
            pred_index = 0  # single sigmoid neuron
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap_on_image(pil_img, heatmap, alpha=0.35, colormap=cv2.COLORMAP_JET):
    """Overlay heatmap onto original image (PIL -> PIL)."""
    img = pil_img.convert("RGB")
    img_arr = np.array(img)

    heatmap_resized = cv2.resize(heatmap, (img_arr.shape[1], img_arr.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (1 - alpha) * img_arr + alpha * heatmap_color
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)
