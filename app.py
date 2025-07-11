# app.py
import os
os.environ["STREAMLIT_WATCHED_FILES"] = ""
import streamlit as st
import numpy as np
import cv2
import joblib
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model

# Load EfficientNet model (without top layer for feature extraction)
base_model = EfficientNetB0(include_top=False, pooling='avg', weights='imagenet')
feature_model = Model(inputs=base_model.input, outputs=base_model.output)

# Load trained classifier and labels
rf_model, class_names = joblib.load("rf_parts_classifier.pkl")

# UI
st.title(" Machine Part Classifier (EfficientNet + RF)")
st.write("Upload a machine part image (bolt, nut, washer, locatingpin.)")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Resize and preprocess for EfficientNet
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized.astype(np.float32), axis=0)
    img_array = preprocess_input(img_array)

    # Extract features using EfficientNet
    features = feature_model.predict(img_array)
    features_flat = features.reshape(1, -1)  # shape (1, 1280)

    # Predict using RandomForest
    prediction = rf_model.predict(features_flat)[0]
    predicted_label = class_names[prediction]

    st.success(f" Predicted Part: **{predicted_label}**")