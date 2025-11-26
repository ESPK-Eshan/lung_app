# ==========================================================
# Lung Disease Classification Streamlit App (FINAL VERSION)
# Includes:
#   - DenseNet121, EfficientNet-B0, ResNet50, Ensemble
#   - TTA, Image Quality Check, CLAHE Enhancement
#   - Confidence Reliability, Disease Info
#   - Probability Chart + Table
#   - Model Comparison Page
#   - Training Artifacts Page
#   - Batch Prediction Page
#   - Medical-Style AI PDF Report Generator
# ==========================================================

import os
import time
from pathlib import Path
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import timm
import cv2
import pandas as pd
from fpdf import FPDF


# ----------------------------
# CONFIG
# ----------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

CLASS_NAMES = [
    "Bacterial Pneumonia",
    "Corona Virus Disease",
    "Normal",
    "Tuberculosis",
    "Viral Pneumonia",
]
NUM_CLASSES = len(CLASS_NAMES)

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ----------------------------
# Disease Info Dictionary
# ----------------------------
DISEASE_INFO = {
    "Bacterial Pneumonia":
        "A bacterial infection causing fluid and inflammation in lung air sacs. "
        "Symptoms include fever, productive cough, chest pain, and breathing difficulty. "
        "Seek medical care immediately if symptoms worsen.",

    "Corona Virus Disease":
        "COVID-19 primarily affects the respiratory system. "
        "Common symptoms include persistent dry cough, fever, chest tightness, and fatigue. "
        "Monitor oxygen saturation and seek help if breathing worsens.",

    "Normal":
        "This X-ray shows no visible abnormalities related to pneumonia, COVID-19, or TB. "
        "However, symptoms should still be evaluated by a clinician if persistent.",

    "Tuberculosis":
        "TB is a chronic bacterial infection damaging lung tissue. "
        "Symptoms include chronic cough, weight loss, fatigue, and night sweats. "
        "Requires long-term medication adherence.",

    "Viral Pneumonia":
        "Inflammation of the lungs caused by viral infection. "
        "Symptoms include cough, fever, chills, and shortness of breath. "
        "Supportive care and hydration are critical."
}


# ----------------------------
# STREAMLIT PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Lung Disease Diagnosis",
    page_icon="🫁",
    layout="wide"
)

st.title("🫁 Lung Disease Classification — Deep Learning Medical AI System")
st.caption("ESHAN PURI (8025340013) | M.E. - AI Project | PyTorch Models | Ensemble | Quality Filters | Medical PDF Report Generator")


# ----------------------------
# MODEL LOADING
# ----------------------------
@st.cache_resource(show_spinner="Loading model architectures...")
def load_backbones():
    return {
        "densenet121": timm.create_model("densenet121", pretrained=False, num_classes=NUM_CLASSES),
        "efficientnet_b0": timm.create_model("efficientnet_b0", pretrained=False, num_classes=NUM_CLASSES),
        "resnet50": timm.create_model("resnet50", pretrained=False, num_classes=NUM_CLASSES),
    }


@st.cache_resource(show_spinner="Loading trained weights...")
def load_models():
    backbones = load_backbones()
    models = {}

    def load_ckpt(name, filename):
        path = MODELS_DIR / filename
        if not path.exists():
            st.warning(f"⚠ Missing model file: {filename}")
            return None

        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            ckpt = ckpt["model_state"]

        model = backbones[name]
        model.load_state_dict(ckpt)
        model.eval()
        return model

    models["densenet121"] = load_ckpt("densenet121", "densenet121.pth")
    models["efficientnet_b0"] = load_ckpt("efficientnet_b0", "efficientnet_b0.pth")
    models["resnet50"] = load_ckpt("resnet50", "resnet50.pth")

    return models


MODELS = load_models()


# ----------------------------
# IMAGE QUALITY FUNCTIONS
# ----------------------------
def compute_blur_score(pil_img):
    img = np.array(pil_img.convert("L"))
    return cv2.Laplacian(img, cv2.CV_64F).var()

def brightness_contrast(pil_img):
    img = np.array(pil_img.convert("L"))
    return img.mean(), img.std()

def enhance_xray(pil_img):
    # Convert to grayscale
    img = np.array(pil_img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img)

    # 🔥 Convert back to 3-channel RGB for model compatibility
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(enhanced_rgb)



# ----------------------------
# PREDICTION FUNCTIONS
# ----------------------------
def preprocess(pil_img):
    return val_tfms(pil_img).unsqueeze(0)

def predict_probs(model, pil_img):
    x = preprocess(pil_img)
    with torch.no_grad():
        logits = model(x)
        return F.softmax(logits, dim=1).cpu().numpy()[0]

def tta_predict(model, pil_img):
    imgs = [pil_img, pil_img.transpose(Image.FLIP_LEFT_RIGHT)]
    probs = [predict_probs(model, im) for im in imgs]
    return np.mean(np.stack(probs), axis=0)

def ensemble_predict(pil_img, tta=False):
    selected = []
    for m in ["densenet121", "efficientnet_b0"]:
        if MODELS[m] is not None:
            selected.append(MODELS[m])

    if len(selected) == 0:
        return np.zeros(NUM_CLASSES)

    probs = []
    for model in selected:
        if tta:
            probs.append(tta_predict(model, pil_img))
        else:
            probs.append(predict_probs(model, pil_img))

    return np.mean(np.stack(probs), axis=0)

def predict_single_image(pil_img, model_choice, enhance=False, tta_enabled=False):
    """
    Unified wrapper so the UI can call a single function for prediction.
    """

    # Step 1 — Image Quality Checks
    blur = compute_blur_score(pil_img)
    bright, contr = brightness_contrast(pil_img)

    # Step 2 — Optional Enhancement
    if enhance:
        pil_img = enhance_xray(pil_img)

    # Step 3 — Choose Model
    if model_choice == "Ensemble":
        probs = ensemble_predict(pil_img, tta_enabled)
    else:
        model = MODELS[model_choice]
        if model is None:
            return None, None, None, blur, bright, contr

        # TTA or normal prediction
        if tta_enabled:
            probs = tta_predict(model, pil_img)
        else:
            probs = predict_probs(model, pil_img)

    # Normalize
    probs /= probs.sum()

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    conf = probs[pred_idx] * 100

    return pred_class, conf, probs, blur, bright, contr

# ----------------------------
# PDF REPORT GENERATOR
# ----------------------------
def generate_pdf_report(
    pil_img, pred_class, conf, probs,
    blur, bright, contr,
    disease_info, model_used
):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "AI-Based Lung Disease Diagnosis Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, "Generated by ESHAN PURI (8025340013) | M.E. - AI Project", ln=True, align="C")
    pdf.ln(8)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Patient Information", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Patient ID: AI-{int(time.time())}", ln=True)
    pdf.cell(0, 8, f"Report Date: {time.strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(6)

    temp_img = "temp_image.png"
    pil_img.save(temp_img)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Uploaded X-ray Image:", ln=True)
    pdf.image(temp_img, w=80)
    pdf.ln(8)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "AI Prediction Result", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Disease: {pred_class}", ln=True)
    pdf.cell(0, 8, f"Confidence: {conf:.2f}%", ln=True)
    pdf.cell(0, 8, f"Model Used: {model_used}", ln=True)
    pdf.ln(6)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Probability Distribution", ln=True)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(80, 8, "Condition", border=1)
    pdf.cell(40, 8, "Probability (%)", border=1, ln=True)

    pdf.set_font("Arial", "", 12)
    for c, p in zip(CLASS_NAMES, probs):
        pdf.cell(80, 8, c, border=1)
        pdf.cell(40, 8, f"{p*100:.2f}", border=1, ln=True)

    pdf.ln(6)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Image Quality Assessment", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Blur Score: {blur:.2f}", ln=True)
    pdf.cell(0, 8, f"Brightness: {bright:.2f}", ln=True)
    pdf.cell(0, 8, f"Contrast: {contr:.2f}", ln=True)
    pdf.ln(6)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"About {pred_class}", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, disease_info)
    pdf.ln(6)

    pdf.set_font("Arial", "I", 10)
    pdf.set_text_color(150, 0, 0)
    pdf.multi_cell(0, 7,
        "Disclaimer: This is an AI-assisted analysis and NOT a medical diagnosis. "
        "Please consult a certified radiologist for clinical confirmation."
    )

    return pdf.output(dest="S").encode("latin1")


# ----------------------------
# PLOTTING
# ----------------------------
def plot_probs(probs):
    fig, ax = plt.subplots(figsize=(5,2.5))
    ax.barh(CLASS_NAMES, probs, color='gray')
    ax.invert_yaxis()
    ax.set_xlabel("Probability")
    plt.tight_layout()
    return fig

def show_probability_chart(probs):
    """
    Wrapper to display probability distribution using the existing plot_probs() function.
    """
    fig = plot_probs(probs)
    st.pyplot(fig)


# ----------------------------
# SIDEBAR NAVIGATION
# ----------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966489.png", width=100)
    st.markdown(
        "<h3 style='text-align:center;'>Navigation</h3>",
        unsafe_allow_html=True
    )

    page = st.radio(
        "Go to:",
        ["Home", "Single Prediction", "Model Evaluation", "Batch Prediction"],
        index=0
    )

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align:center; color:#888;'>
            <small>🫁 Lung Disease AI System<br>Developed by <b>Eshan Puri</b></small>
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------------------
# HOME PAGE — Modern UI
# ----------------------------
if page == "Home":

    st.markdown(
        """
        <h1 style='text-align:center; color:#00b4d8;'>
            🫁 Lung Disease Classification System
        </h1>
        <p style='text-align:center; font-size:18px;'>
            AI-powered chest X-ray analysis with multi-model inference, image quality checks,
            medical-style PDF reporting, and evaluation dashboard.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # THREE FEATURE CARDS
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🔍 Single Image Diagnosis")
        st.write(
            """
            • Upload X-ray  
            • Enhance image  
            • AI-based disease prediction  
            • Confidence + probability chart  
            • Medical explanation  
            • Download PDF report  
            """
        )

    with col2:
        st.markdown("### 📊 Evaluation Dashboard")
        st.write(
            """
            • Model comparison  
            • Confusion matrix  
            • ROC curves  
            • Training metrics  
            """
        )

    with col3:
        st.markdown("### 📁 Batch Processing")
        st.write(
            """
            • Upload multiple X-rays  
            • AI prediction for each  
            • Export results as CSV  
            • Fast automated processing  
            """
        )

    st.markdown("---")

    st.markdown(
        """
        <p style='text-align:center; color:#888;'>
            Built using <b>Python</b>, <b>Streamlit</b>, <b>PyTorch</b>, <b>TIMM</b>, <b>OpenCV</b>, <b>Matplotlib</b>, <b>Pandas</b>.
        </p>
        """,
        unsafe_allow_html=True
    )

# ----------------------------
# PAGE 1 — SINGLE PREDICTION
# ----------------------------
elif page == "Single Prediction":

    st.markdown("## 🔍 Single X-Ray Diagnosis")
    st.write("Upload an X-ray image to generate disease prediction and medical report.")

    uploaded_file = st.file_uploader("Upload Chest X-ray", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        pil_img = Image.open(uploaded_file).convert("RGB")

        st.markdown("### 📸 Preview")
        st.image(pil_img, width=350)

        enhance = st.checkbox("Enhance Image (CLAHE)", value=False)
        tta_enabled = st.checkbox("Use Test-Time Augmentation (TTA)", value=False)

        model_choice = st.selectbox(
            "Choose Model",
            ["densenet121", "efficientnet_b0", "resnet50", "Ensemble"],
            index=0
        )

        if st.button("🚀 Run Diagnosis"):
            with st.spinner("Analyzing X-ray with AI models..."):
                pred_class, conf, probs, blur, bright, contr = predict_single_image(
                    pil_img,
                    model_choice,
                    enhance,
                    tta_enabled
                )

            # RESULTS CARD
            st.markdown(
                f"""
                <div style="padding:15px; border-radius:10px; background-color:#1c1e29;">
                    <h3 style="color:#00b4d8;">Prediction Summary</h3>
                    <h4>🩺 Disease: <b style="color:#ff6b6b;">{pred_class}</b></h4>
                    <p><b>Confidence:</b> {conf:.2f}%</p>
                    <p><b>Model Used:</b> {model_choice}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### 📊 Probability Distribution")
            show_probability_chart(probs)

            st.markdown("### 🧪 Image Quality Checks")
            st.write(f"Blur score: **{blur:.2f}**")
            st.write(f"Brightness: **{bright:.2f}**")
            st.write(f"Contrast: **{contr:.2f}**")

            st.markdown("### 📄 Download Medical Report")
            st.download_button(
                label="📄 Download PDF Report",
                data=generate_pdf_report(
                    pil_img, pred_class, conf, probs,
                    blur, bright, contr,
                    DISEASE_INFO[pred_class],
                    model_choice
                ),
                file_name=f"lung_report_{pred_class}.pdf",
                mime="application/pdf"
            )


# ----------------------------
# PAGE 2 — MODEL EVALUATION
# ----------------------------
elif page == "Model Evaluation":

    st.header("Model Evaluation & Metrics")

    # Model options for dropdown
    model_options = ["densenet121", "efficientnet_b0", "resnet50", "Ensemble"]
    selected_model = st.selectbox("Select model to view evaluation plots", model_options)

    st.subheader(f"Selected Model: {selected_model}")

    # MODEL ACCURACY TABLE
    st.subheader("Model Comparison Table")
    model_metrics = {
        "densenet121": {"Accuracy": "0.84"},
        "efficientnet_b0": {"Accuracy": "0.83"},
        "resnet50": {"Accuracy": "0.81"},
        "Ensemble (Top-2)": {"Accuracy": "0.855"},
    }
    st.table(pd.DataFrame(model_metrics).T)

    # ----------------------------
    # Confusion Matrix & ROC paths (REVERSE NAMING)
    # ----------------------------
    if selected_model != "Ensemble":
        cm_path = RESULTS_DIR / f"{selected_model}_confusion_matrix.png"
        roc_path = RESULTS_DIR / f"{selected_model}_roc_curve.png"
    else:
        cm_path = RESULTS_DIR / "ensemble_confusion_matrix.png"
        roc_path = RESULTS_DIR / "ensemble_roc_curve.png"

    # ----------------------------
    # SHOW CONFUSION MATRIX
    # ----------------------------
    st.subheader("Confusion Matrix")
    if cm_path.exists():
        st.image(str(cm_path), caption=f"Confusion Matrix — {selected_model}")
    else:
        st.warning(f"No confusion matrix found for {selected_model} at {cm_path}")

    # ----------------------------
    # SHOW ROC CURVE
    # ----------------------------
    st.subheader("ROC Curve")
    if roc_path.exists():
        st.image(str(roc_path), caption=f"ROC Curve — {selected_model}")
    else:
        st.warning(f"No ROC curve found for {selected_model} at {roc_path}")


# ----------------------------
# PAGE 3 — BATCH PREDICTION
# ----------------------------
else:
    st.header("Batch Prediction")

    files = st.file_uploader(
        "Upload Multiple X-rays",
        type=['png','jpg','jpeg'],
        accept_multiple_files=True
    )

    if files:
        results = []
        for f in files:
            pil_img = Image.open(f).convert("RGB")
            probs = ensemble_predict(pil_img)
            pred = CLASS_NAMES[int(np.argmax(probs))]
            conf = probs.max()*100
            results.append([f.name, pred, f"{conf:.1f}%"])

        df = pd.DataFrame(results, columns=["Filename", "Prediction", "Confidence"])
        st.table(df)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "batch_predictions.csv"
        )
