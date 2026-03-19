import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import time

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="CoffeeGuard",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2d5a27;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 1rem;
        color: #6b7280;
        margin-top: 0;
    }
    .metric-card {
        background: #f0f7ee;
        border-radius: 12px;
        padding: 1rem;
        border-left: 4px solid #2d5a27;
    }
    .stButton > button {
        background-color: #2d5a27;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #4a8c3f;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## CoffeeGuard")
    st.markdown("Arabica Coffee Leaf Disease Classifier")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Dashboard", "Predict", "Retrain"],
        label_visibility="collapsed"
    )

    st.divider()

    try:
        res = requests.get(f"{API_URL}/health", timeout=3)
        if res.status_code == 200:
            st.success("API Online")
        else:
            st.error("API Offline")
    except:
        st.error("API Offline")

    try:
        res = requests.get(f"{API_URL}/metrics", timeout=3)
        if res.status_code == 200:
            data = res.json()
            st.markdown(f"**Uptime:** {data['uptime']}")
    except:
        pass

# Dataset stats
CLASSES = ["Cerscospora", "Healthy", "Leaf rust", "Miner", "Phoma"]
CLASS_COUNTS = [7681, 18983, 8336, 16978, 6571]
F1_SCORES = [0.98, 1.00, 0.96, 0.99, 0.99]
PRECISION = [0.96, 1.00, 1.00, 0.99, 0.98]
RECALL = [1.00, 1.00, 0.92, 1.00, 1.00]
SEVERITY = ["Moderate", "None", "High", "Moderate", "Moderate"]
SEVERITY_COLORS = {"None": "#4ade80", "Moderate": "#f97316", "High": "#ef4444"}

# ─────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────
if page == "Dashboard":
    st.markdown('<p class="main-title">CoffeeGuard Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Arabica Coffee Leaf Disease Classification — Model Performance Overview</p>', unsafe_allow_html=True)
    st.divider()

    # Metric cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Overall Accuracy", "98.8%", "On 11,713 val images")
    with col2:
        st.metric("Macro F1 Score", "98.4%", "Balanced across 5 classes")
    with col3:
        st.metric("Macro Precision", "98.5%", "Low false positive rate")
    with col4:
        st.metric("Macro Recall", "98.3%", "Low false negative rate")
    with col5:
        st.metric("Training Images", "58,549", "Arabica leaves, East Africa")

    st.divider()

    # Visualization 1 — Dataset Distribution
    st.subheader("Feature 1 — Dataset Class Distribution")
    st.markdown("""
    The dataset is imbalanced — Healthy leaves make up 32% of all images while Phoma 
    makes up only 11%. This reflects the real-world distribution of disease prevalence 
    on Arabica coffee farms in East Africa. Healthy leaves are more common than diseased 
    ones, which is expected. Phoma and Cerscospora are less common diseases, which is 
    why the dataset has fewer examples. This imbalance was handled during training using 
    class weights.
    """)

    df_dist = pd.DataFrame({
        "Class": CLASSES,
        "Images": CLASS_COUNTS,
        "Severity": SEVERITY
    })

    fig1 = px.bar(
        df_dist,
        x="Class",
        y="Images",
        color="Severity",
        color_discrete_map=SEVERITY_COLORS,
        text="Images",
        title="Number of Images per Disease Class"
    )
    fig1.update_traces(textposition="outside")
    fig1.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="#1a1a1a",
        height=400
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.divider()

    # Visualization 2 — Per-Class Evaluation Metrics
    st.subheader("Feature 2 — Per-Class Evaluation Metrics")
    st.markdown("""
    Precision, Recall and F1 Score tell different stories per class. Leaf rust has 
    perfect precision (1.00) but lower recall (0.92) — meaning when the model predicts 
    Leaf rust it is always right, but it misses 8% of actual Leaf rust cases, 
    misclassifying them as Cerscospora due to visual similarity between early-stage 
    rust spots and cercospora lesions. Every other class achieves near-perfect scores 
    across all three metrics.
    """)

    df_metrics = pd.DataFrame({
        "Class": CLASSES * 3,
        "Score": PRECISION + RECALL + F1_SCORES,
        "Metric": ["Precision"] * 5 + ["Recall"] * 5 + ["F1 Score"] * 5
    })

    fig2 = px.bar(
        df_metrics,
        x="Class",
        y="Score",
        color="Metric",
        barmode="group",
        title="Precision, Recall and F1 Score per Disease Class",
        color_discrete_sequence=["#2d5a27", "#4a8c3f", "#a8d5a0"]
    )
    fig2.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="#1a1a1a",
        yaxis_range=[0.85, 1.02],
        height=400
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # Visualization 3 — Disease Severity Distribution
    st.subheader("Feature 3 — Disease Severity Distribution")
    st.markdown("""
    Not all diseases are equally dangerous. Leaf rust is classified as High severity 
    because it can destroy up to 70% of a harvest if not treated immediately. 
    Cerscospora, Miner, and Phoma are Moderate — damaging but manageable with timely 
        intervention. Healthy leaves need no action. This severity mapping gives farmers 
    an immediate action priority rather than just a class label, making the tool 
    practically useful in the field.
    """)

    severity_counts = pd.DataFrame({
        "Severity": ["None", "Moderate", "High"],
        "Classes": [1, 3, 1],
        "Example": ["Healthy", "Cerscospora, Miner, Phoma", "Leaf rust"]
    })

    fig3 = px.pie(
        severity_counts,
        values="Classes",
        names="Severity",
        title="Disease Classes by Severity Level",
        color="Severity",
        color_discrete_map=SEVERITY_COLORS,
        hole=0.4
    )
    fig3.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="#1a1a1a",
        height=400
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # Model architecture info
    st.subheader("Model Architecture")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        | Parameter | Value |
        |---|---|
        | Base model | MobileNetV2 (ImageNet) |
        | Training | Two-phase transfer learning |
        | Total parameters | 2,620,997 |
        | Input size | 224 × 224 × 3 |
        | Optimizer | Adam, lr=0.0001 |
        | Regularization | L2 + Dropout |
        """)
    with col2:
        st.markdown("""
        | Metric | Score |
        |---|---|
        | Best val accuracy | 99.33% |
        | Overall accuracy | 98.80% |
        | Macro F1 | 98.36% |
        | Macro Precision | 98.50% |
        | Macro Recall | 98.31% |
        """)

# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
elif page == "Predict":
    st.markdown('<p class="main-title">Diagnose a Coffee Leaf</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Upload a clear photo of a single Arabica coffee leaf to get an instant disease diagnosis.</p>', unsafe_allow_html=True)
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload a leaf image",
        type=["jpg", "jpeg", "png"],
        help="Take a clear photo of a single leaf in good lighting"
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])

        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded leaf", use_column_width=True)

        with col2:
            with st.spinner("Analyzing leaf..."):
                try:
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    res = requests.post(f"{API_URL}/predict", files=files, timeout=30)

                    if res.status_code == 200:
                        result = res.json()

                        predicted = result["predicted_class"]
                        confidence = result["confidence"]
                        latency = result["latency_ms"]
                        info = result["disease_info"]
                        probs = result["all_probabilities"]

                        severity_color = {
                            "None": "green",
                            "Moderate": "orange",
                            "High": "red"
                        }.get(info["severity"], "gray")

                        st.markdown(f"### {predicted}")
                        st.markdown(f"**Confidence:** {confidence}% &nbsp;|&nbsp; **Latency:** {latency}ms")
                        st.markdown(f"**Severity:** :{severity_color}[{info['severity']}]")
                        st.divider()
                        st.markdown(f"**Description:** {info['description']}")
                        st.markdown(f"**Recommended action:** {info['action']}")
                        st.divider()

                        st.markdown("**Probability distribution:**")
                        df_probs = pd.DataFrame({
                            "Class": list(probs.keys()),
                            "Probability (%)": list(probs.values())
                        }).sort_values("Probability (%)", ascending=True)

                        fig = px.bar(
                            df_probs,
                            x="Probability (%)",
                            y="Class",
                            orientation="h",
                            color="Probability (%)",
                            color_continuous_scale=["#e8f0e6", "#2d5a27"],
                            title="Confidence per class"
                        )
                        fig.update_layout(
                            plot_bgcolor="white",
                            paper_bgcolor="white",
                            font_color="#1a1a1a",
                            height=280,
                            showlegend=False,
                            coloraxis_showscale=False
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        st.error(f"Prediction failed: {res.text}")

                except Exception as e:
                    st.error(f"Could not connect to API: {str(e)}")

# ─────────────────────────────────────────────
# RETRAIN
# ─────────────────────────────────────────────
elif page == "Retrain":
    st.markdown('<p class="main-title">Upload Data & Retrain</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Upload new labeled leaf images to improve the model, then trigger retraining.</p>', unsafe_allow_html=True)
    st.divider()

    st.subheader("Step 1 — Upload Images")

    label = st.selectbox(
        "Select disease class for uploaded images",
        ["", "Cerscospora", "Healthy", "Leaf rust", "Miner", "Phoma"],
        index=0
    )

    uploaded_files = st.file_uploader(
        "Upload leaf images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload multiple images for the selected class"
    )

    if st.button("Upload Images") and label and uploaded_files:
        with st.spinner(f"Uploading {len(uploaded_files)} images..."):
            try:
                files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
                res = requests.post(
                    f"{API_URL}/upload?label={label}",
                    files=files,
                    timeout=60
                )
                if res.status_code == 200:
                    data = res.json()
                    st.success(data["message"])
                    st.json(data)
                else:
                    st.error(f"Upload failed: {res.text}")
            except Exception as e:
                st.error(f"Could not connect to API: {str(e)}")
    elif st.button("Upload Images"):
        if not label:
            st.warning("Please select a disease class first")
        if not uploaded_files:
            st.warning("Please select at least one image")

    st.divider()
    st.subheader("Step 2 — Trigger Retraining")
    st.markdown("""
    Once images are uploaded, click the button below to retrain the model.
    Retraining runs in the background — check status after triggering.
    """)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Retrain Model", type="primary"):
            with st.spinner("Triggering retraining..."):
                try:
                    res = requests.post(f"{API_URL}/retrain", timeout=10)
                    if res.status_code == 200:
                        st.success(res.json()["message"])
                    elif res.status_code == 409:
                        st.warning("Retraining is already in progress")
                    else:
                        st.error(f"Failed: {res.text}")
                except Exception as e:
                    st.error(f"Could not connect to API: {str(e)}")

    with col2:
        if st.button("Check Status"):
            try:
                res = requests.get(f"{API_URL}/retrain/status", timeout=5)
                if res.status_code == 200:
                    data = res.json()
                    st.json(data)
                    if data["is_training"]:
                        st.info("Retraining in progress...")
                    elif data["last_status"] == "completed":
                        st.success(f"Last retraining completed. Accuracy: {data['last_accuracy']}%")
                    elif data["last_status"] == "idle":
                        st.info("No retraining has been triggered yet")
                    else:
                        st.warning(f"Status: {data['last_status']}")
            except Exception as e:
                st.error(f"Could not connect to API: {str(e)}")
