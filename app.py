"""DigitAI Studio: Draw & Predict, Beat the AI, Analytics, CNN Explorer."""
from pathlib import Path
import json
import random

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf

from predict import load_cnn, load_mlp, preprocess_image, preprocess_canvas, CNN_PATH, MLP_PATH

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports"

st.set_page_config(page_title="DigitAI Studio | NNDL", page_icon="🔢", layout="wide")

# Compact custom styling
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px;}
.hero {padding: 1.25rem 1.5rem; border-radius: 18px; background: linear-gradient(120deg,#21164b,#152b50,#123d46); margin-bottom: 1rem;}
.hero h1 {margin:0; color:#fff; font-size:2.25rem;}
.hero p {color:#d8defa; margin:.45rem 0 0 0;}
div[data-testid="stMetric"] {background:rgba(120,110,180,.12); padding:12px; border-radius:12px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🔢 DigitAI Studio</h1><p>NNDL Mini Project · Draw, Predict, Play and Explore a CNN</p></div>', unsafe_allow_html=True)

if not CNN_PATH.exists():
    st.warning("No trained model found. Open the project terminal and run `python train.py` first.")
    st.stop()

@st.cache_resource
def get_cnn():
    return load_cnn()

@st.cache_resource
def get_mlp():
    return load_mlp()

cnn = get_cnn()
tabs = st.tabs(["✍️ Draw & Predict", "🎮 Beat the AI", "📊 Analytics Lab", "🧠 CNN Explorer"])

def show_probabilities(probabilities):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar([str(i) for i in range(10)], probabilities)
    ax.set_xlabel("Digit")
    ax.set_ylabel("Model score")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

with tabs[0]:
    st.subheader("Draw a digit or upload a photo")
    st.write("Draw one large digit using your mouse or touchscreen. For best results, draw dark ink on a light canvas.")
    left, right = st.columns([1.1, 1])
    with left:
        canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=16,
    stroke_color="#111111",
    background_color="#FFFFFF",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="digit_canvas",
    return_image_data=True,
)

        predict_draw = st.button("✨ Predict my drawing", type="primary", use_container_width=True)
    with right:
        uploaded = st.file_uploader("Or upload a digit photo", type=["png", "jpg", "jpeg"], key="photo_upload")
        predict_upload = st.button("🔍 Predict uploaded image", use_container_width=True, disabled=uploaded is None)

    if predict_draw:
        if canvas_result.image_data is None or not np.any(canvas_result.image_data[:, :, :3] < 245):
            st.info("Draw a digit on the canvas first.")
        else:
            try:
                tensor = preprocess_canvas(canvas_result.image_data)
                probs = cnn.predict(tensor, verbose=0)[0]
                pred = int(np.argmax(probs))
                st.success(f"The CNN predicts **{pred}**")
                a, b = st.columns(2)
                a.metric("Predicted digit", str(pred))
                b.metric("Top score", f"{probs[pred] * 100:.2f}%")
                show_probabilities(probs)
                with st.expander("See the 28×28 image sent to the CNN"):
                    st.image(tensor[0, :, :, 0], clamp=True, caption="Preprocessed model input")
            except Exception as exc:
                st.error(f"Could not process drawing: {exc}")

    if predict_upload and uploaded is not None:
        try:
            image = Image.open(uploaded)
            tensor = preprocess_image(image)
            probs = cnn.predict(tensor, verbose=0)[0]
            pred = int(np.argmax(probs))
            a, b = st.columns(2)
            with a:
                st.image(image, caption="Uploaded image", use_container_width=True)
            with b:
                st.metric("Predicted digit", str(pred))
                st.metric("Top score", f"{probs[pred] * 100:.2f}%")
            show_probabilities(probs)
            with st.expander("See the 28×28 preprocessed input"):
                st.image(tensor[0, :, :, 0], clamp=True)
        except Exception as exc:
            st.error(f"Could not process image: {exc}")

with tabs[1]:
    st.subheader("Beat the AI")
    st.write("Choose what digit you think is shown. Then reveal the answer and compare it with the CNN.")
    @st.cache_data
    def load_game_data():
        (xtr, ytr), (xte, yte) = tf.keras.datasets.mnist.load_data()
        return xte, yte
    try:
        x_game, y_game = load_game_data()
        if "game_index" not in st.session_state:
            st.session_state.game_index = random.randrange(len(y_game))
        if "game_revealed" not in st.session_state:
            st.session_state.game_revealed = False
        if "game_score" not in st.session_state:
            st.session_state.game_score = 0
        if "game_rounds" not in st.session_state:
            st.session_state.game_rounds = 0

        idx = st.session_state.game_index
        game_cols = st.columns([1, 1.2])
        with game_cols[0]:
            st.image(x_game[idx], caption="What digit is this?", width=240)
        with game_cols[1]:
            guess = st.selectbox("Your guess", list(range(10)), key=f"guess_{idx}")
            if st.button("Reveal answer", type="primary", key=f"reveal_{idx}"):
                if not st.session_state.game_revealed:
                    if int(guess) == int(y_game[idx]):
                        st.session_state.game_score += 1
                    st.session_state.game_rounds += 1
                    st.session_state.game_revealed = True
                st.rerun()
            if st.session_state.game_revealed:
                actual = int(y_game[idx])
                sample = x_game[idx].astype("float32")[None, ..., None] / 255.0
                model_pred = int(np.argmax(cnn.predict(sample, verbose=0)[0]))
                st.write(f"**Actual digit:** {actual}")
                st.write(f"**CNN prediction:** {model_pred}")
                st.write("You got it! 🎉" if int(guess) == actual else "Not this time — try another round!")
                st.write("The CNN was correct too! 🤖" if model_pred == actual else "The CNN missed this one.")
        m1, m2 = st.columns(2)
        m1.metric("Your score", f"{st.session_state.game_score}/{st.session_state.game_rounds}")
        m2.metric("Rounds played", st.session_state.game_rounds)
        if st.button("➡️ Next digit", key="next_game"):
            st.session_state.game_index = random.randrange(len(y_game))
            st.session_state.game_revealed = False
            st.rerun()
        if st.button("🔄 Reset score", key="reset_game"):
            st.session_state.game_score = 0
            st.session_state.game_rounds = 0
            st.session_state.game_index = random.randrange(len(y_game))
            st.session_state.game_revealed = False
            st.rerun()
    except Exception as exc:
        st.error(f"Could not load game dataset: {exc}")

with tabs[2]:
    st.subheader("Model performance dashboard")
    metrics_path = REPORT_DIR / "metrics.json"
    if not metrics_path.exists():
        st.info("Run `python train.py` to generate evaluation metrics and charts.")
    else:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        c1, c2, c3 = st.columns(3)
        c1.metric("CNN test accuracy", f"{metrics['cnn_test_accuracy'] * 100:.2f}%")
        c2.metric("MLP test accuracy", f"{metrics['mlp_test_accuracy'] * 100:.2f}%")
        c3.metric("CNN test loss", f"{metrics['cnn_test_loss']:.4f}")
        st.caption("Metrics are from the held-out MNIST test set and your latest training run.")
        curve = REPORT_DIR / "training_curves.png"
        matrix = REPORT_DIR / "confusion_matrix.png"
        c1, c2 = st.columns(2)
        if curve.exists():
            c1.image(str(curve), caption="CNN training and validation curves", use_container_width=True)
        if matrix.exists():
            c2.image(str(matrix), caption="CNN confusion matrix", use_container_width=True)
        report_path = REPORT_DIR / "classification_report.txt"
        if report_path.exists():
            with st.expander("View classification report"):
                st.code(report_path.read_text(encoding="utf-8"), language="text")

with tabs[3]:
    st.subheader("How does the CNN work?")
    st.write("A CNN learns local visual patterns (edges and strokes) before combining them to classify a digit.")
    architecture = [
        ("Input", "28 × 28 × 1 grayscale image"),
        ("Conv2D — 32 filters", "Learns low-level patterns such as edges and curves"),
        ("MaxPooling2D", "Reduces spatial dimensions while retaining useful features"),
        ("Conv2D — 64 filters", "Learns more complex digit patterns"),
        ("MaxPooling2D", "Further reduces feature-map size"),
        ("Flatten", "Converts feature maps into a vector"),
        ("Dense — 128 units", "Combines learned features"),
        ("Dropout — 30%", "Regularization to reduce overfitting"),
        ("Dense — 10 + Softmax", "Scores the ten digit classes (0–9)"),
    ]
    for layer, explanation in architecture:
        with st.container(border=True):
            st.markdown(f"**{layer}**")
            st.caption(explanation)
    st.divider()
    st.subheader("CNN vs. MLP baseline")
    st.write("The CNN uses spatial filters and pooling; the MLP flattens pixels and learns dense connections.")
    if MLP_PATH.exists():
        try:
            mlp = get_mlp()
            if (REPORT_DIR / "metrics.json").exists():
                metrics = json.loads((REPORT_DIR / "metrics.json").read_text(encoding="utf-8"))
                x1, x2 = st.columns(2)
                x1.metric("CNN test accuracy", f"{metrics['cnn_test_accuracy']*100:.2f}%")
                x2.metric("MLP test accuracy", f"{metrics['mlp_test_accuracy']*100:.2f}%")
                st.caption("Use the measured results from your own run in your project report.")
            with st.expander("Show CNN layer summary"):
                lines = []
                cnn.summary(print_fn=lambda line: lines.append(line))
                st.code("\n".join(lines))
            with st.expander("Show MLP layer summary"):
                lines = []
                mlp.summary(print_fn=lambda line: lines.append(line))
                st.code("\n".join(lines))
        except Exception as exc:
            st.warning(f"Could not load comparison model: {exc}")
    else:
        st.info("Train both models by running `python train.py` to enable the MLP comparison.")

st.divider()
st.caption("DigitAI Studio · Educational NNDL prototype. Model scores can be wrong, especially on photos that differ from MNIST.")
