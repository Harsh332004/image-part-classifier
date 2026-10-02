"""Streamlit web app for the machine-part classifier."""
import json

import pandas as pd
import streamlit as st

from common import METRICS_PATH, load_artifacts, load_rgb, predict_proba

st.set_page_config(page_title="Machine Part Classifier", page_icon="🔩")


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_artifacts()          # loaded once, not on every rerun


st.title("🔩 Machine Part Classifier")

try:
    model, class_names = get_model()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()

st.write(f"Upload a photo of a part. Supported classes: **{', '.join(class_names)}**.")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Minimum confidence", 0.30, 0.95, 0.60, 0.05,
                          help="Below this the app says 'uncertain' instead of guessing.")
    use_tta = st.checkbox("Test-time augmentation", value=True,
                          help="Averages predictions over flipped copies (slower, steadier).")
    if METRICS_PATH.exists():
        m = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.header("Model quality")
        st.metric("Test accuracy", f"{m['test_accuracy']:.1%}")
        st.metric("Macro F1", f"{m['macro_f1']:.3f}")
        st.caption(f"Measured on {m['n_test']} images never used in training.")

tab_upload, tab_camera = st.tabs(["Upload", "Camera"])
with tab_upload:
    source = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp", "bmp"])
with tab_camera:
    camera = st.camera_input("Take a photo")
source = source or camera

if source is not None:
    try:
        image = load_rgb(source)
    except Exception:
        st.error("Could not read this file as an image.")
        st.stop()

    st.image(image, caption="Input image", use_container_width=True)

    with st.spinner("Classifying..."):
        probs = predict_proba(model, image, tta=use_tta)

    order = probs.argsort()[::-1]
    best = int(order[0])
    confidence = float(probs[best])

    if confidence >= threshold:
        st.success(f"Predicted part: **{class_names[best]}** ({confidence:.1%})")
    else:
        st.warning(
            f"Uncertain - best guess is **{class_names[best]}** ({confidence:.1%}), "
            "below the confidence threshold. This may not be one of the supported "
            "parts, or the photo may be unclear."
        )

    st.subheader("Class probabilities")
    st.bar_chart(pd.Series({class_names[i]: float(probs[i]) for i in order}))
