from pathlib import Path
import tempfile

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn

from PIL import Image
from torchvision.models import (
    mobilenet_v2,
    MobileNet_V2_Weights
)


# =====================================================
# 1. CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parent

MODEL_DIR = ROOT / "model_outputs_experiment2"

CHECKPOINT = MODEL_DIR / "best_model.pth"
NORMALIZATION = MODEL_DIR / "normalization.npz"

DEVICE = torch.device("cpu")

NUM_FRAMES = 16
IMAGE_SIZE = 224
INPUT_DIM = 1280

LABELS = {
    0: "Normal Behaviour",
    1: "Abnormal Behaviour"
}


# =====================================================
# 2. STREAMLIT PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Video Anomaly Detection",
    page_icon="🎥",
    layout="wide"
)


# =====================================================
# 3. CUSTOM FRONTEND DESIGN
# =====================================================

st.markdown("""
<style>

.main-title {
    font-size: 36px;
    font-weight: 750;
    margin-bottom: 5px;
}

.subtitle {
    color: #9CA3AF;
    font-size: 16px;
    margin-bottom: 25px;
}

.result-card {
    background-color: #20242D;
    padding: 22px;
    border-radius: 12px;
    border: 1px solid #343A46;
}

.result-label {
    color: #A1A1AA;
    font-size: 14px;
}

.result-value {
    font-size: 27px;
    font-weight: 700;
    margin-top: 8px;
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 15px;
}

.footer {
    text-align: center;
    color: #858B98;
    font-size: 13px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 4. HEADER
# =====================================================

st.markdown(
    '<div class="main-title">'
    '🎥 CNN-BiLSTM Video Anomaly Detection'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent surveillance video classification '
    'using MobileNetV2 and Bidirectional LSTM'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =====================================================
# 5. CHECK MODEL FILES
# =====================================================

if not CHECKPOINT.exists():
    st.error("Experiment 2 checkpoint not found.")
    st.stop()

if not NORMALIZATION.exists():
    st.error("Normalization file not found.")
    st.stop()


# =====================================================
# 6. LOAD NORMALIZATION
# =====================================================

@st.cache_resource
def load_normalization():

    data = np.load(NORMALIZATION)

    mean = data["mean"].astype(np.float32)
    std = data["std"].astype(np.float32)

    if mean.shape != (1280,):
        raise ValueError("Invalid normalization mean.")

    if std.shape != (1280,):
        raise ValueError("Invalid normalization std.")

    if not np.isfinite(mean).all():
        raise ValueError("Invalid mean values.")

    if not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid standard deviation.")

    return mean, std


# =====================================================
# 7. LOAD MOBILENETV2
# =====================================================

@st.cache_resource
def load_cnn():

    weights = MobileNet_V2_Weights.DEFAULT

    transform = weights.transforms()

    model = mobilenet_v2(weights=weights)

    model.classifier = nn.Identity()

    model.eval()
    model.to(DEVICE)

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model, transform


# =====================================================
# 8. EXPERIMENT 2 BiLSTM ARCHITECTURE
# =====================================================

class CNNLSTMClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 2)
        )

    def forward(self, x):

        x = self.projection(x)

        output, _ = self.lstm(x)

        x = output.mean(dim=1)

        return self.classifier(x)


# =====================================================
# 9. LOAD TRAINED CLASSIFIER
# =====================================================

@st.cache_resource
def load_classifier():

    model = CNNLSTMClassifier().to(DEVICE)

    state_dict = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    return model


# =====================================================
# 10. INITIALIZE MODELS
# =====================================================

try:

    with st.spinner("Loading deep learning models..."):

        mean, std = load_normalization()

        cnn, transform = load_cnn()

        classifier = load_classifier()

    st.success("Models loaded successfully.")

except Exception as error:

    st.error(f"Model loading failed: {error}")
    st.stop()


# =====================================================
# 11. EXTRACT VIDEO FRAMES
# =====================================================

def extract_frames(video_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("Unable to open video.")

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames < NUM_FRAMES:
        cap.release()
        raise ValueError(
            "Video must contain at least 16 frames."
        )

    indices = np.linspace(
        0,
        total_frames - 1,
        NUM_FRAMES
    ).round().astype(int)

    if len(np.unique(indices)) != NUM_FRAMES:
        cap.release()
        raise ValueError("Unable to sample 16 unique frames.")

    frames = []

    for index in indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(index)
        )

        success, frame = cap.read()

        if not success:
            cap.release()
            raise ValueError(
                f"Unable to read frame {index}."
            )

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(frame)

        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR
        )

        frames.append(image)

    cap.release()

    duration = (
        total_frames / fps if fps > 0 else 0
    )

    return frames, total_frames, duration


# =====================================================
# 12. EXTRACT MOBILENETV2 FEATURES
# =====================================================

def extract_features(frames):

    tensors = [
        transform(frame)
        for frame in frames
    ]

    batch = torch.stack(tensors).to(DEVICE)

    with torch.inference_mode():
        features = cnn(batch)

    features = features.cpu().numpy()

    if features.shape != (16, 1280):
        raise ValueError("Unexpected feature dimensions.")

    if not np.isfinite(features).all():
        raise ValueError("Invalid extracted features.")

    return features.astype(np.float32)


# =====================================================
# 13. VIDEO PREDICTION
# =====================================================

def predict_video(features):

    normalized = (
        features - mean
    ) / std

    x = torch.from_numpy(
        normalized.astype(np.float32)
    ).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():

        logits = classifier(x)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predicted_class = int(
            probabilities.argmax(dim=1).item()
        )

    probabilities = probabilities[0].cpu().numpy()

    confidence = float(
        probabilities[predicted_class]
    )

    return predicted_class, confidence, probabilities


# =====================================================
# 14. VIDEO UPLOAD
# =====================================================

st.markdown(
    '<div class="section-title">Upload Surveillance Video</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a video file",
    type=["mp4", "avi", "mov", "mkv"]
)


# =====================================================
# 15. VIDEO PREVIEW
# =====================================================

if uploaded_file is not None:

    st.subheader("Video Preview")

    st.video(uploaded_file)

    st.caption(
        f"File: {uploaded_file.name} | "
        f"Size: {uploaded_file.size / (1024 * 1024):.2f} MB"
    )

    st.divider()

    analyze = st.button(
        "🔍 Analyze Video",
        type="primary",
        use_container_width=True
    )

    if analyze:

        video_path = None

        try:

            suffix = (
                Path(uploaded_file.name).suffix
                or ".mp4"
            )

            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                delete=False
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getvalue()
                )

                video_path = Path(temp_file.name)

            with st.spinner("Extracting video frames..."):

                frames, total_frames, duration = (
                    extract_frames(video_path)
                )

            with st.spinner("Extracting MobileNetV2 features..."):

                features = extract_features(frames)

            with st.spinner("Running BiLSTM prediction..."):

                predicted_class, confidence, probabilities = (
                    predict_video(features)
                )

            # =========================================
            # RESULTS
            # =========================================

            st.divider()

            st.markdown(
                '<div class="section-title">'
                'Prediction Results'
                '</div>',
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">
                            Predicted Behaviour
                        </div>
                        <div class="result-value">
                            {LABELS[predicted_class]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">
                            Model Confidence
                        </div>
                        <div class="result-value">
                            {confidence * 100:.2f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # =========================================
            # CLASS PROBABILITIES
            # =========================================

            st.markdown(
                '<div class="section-title">'
                'Class Probabilities'
                '</div>',
                unsafe_allow_html=True
            )

            st.write(
                f"🟢 Normal Behaviour — "
                f"{probabilities[0] * 100:.2f}%"
            )

            st.progress(
                float(probabilities[0])
            )

            st.write(
                f"🟠 Abnormal Behaviour — "
                f"{probabilities[1] * 100:.2f}%"
            )

            st.progress(
                float(probabilities[1])
            )

            # =========================================
            # CLASSIFICATION DETAILS
            # =========================================

            st.divider()

            st.markdown(
                '<div class="section-title">'
                'Classification Details'
                '</div>',
                unsafe_allow_html=True
            )

            detail1, detail2 = st.columns(2)

            with detail1:

                st.write(
                    f"**Predicted label:** {predicted_class}"
                )

                st.write(
                    f"**Predicted meaning:** "
                    f"{LABELS[predicted_class]}"
                )

            with detail2:

                st.write(
                    f"**Original video frames:** {total_frames}"
                )

                st.write(
                    f"**Video duration:** {duration:.2f} seconds"
                )

                st.write(
                    f"**Frames used for prediction:** {NUM_FRAMES}"
                )

            # =========================================
            # SAMPLED FRAMES
            # =========================================

            st.divider()

            st.markdown(
                '<div class="section-title">'
                'Sampled Video Frames'
                '</div>',
                unsafe_allow_html=True
            )

            st.write(
                "Sixteen uniformly sampled frames "
                "used by the deep learning model."
            )

            st.image(
                frames,
                caption=[
                    f"Frame {i + 1}"
                    for i in range(NUM_FRAMES)
                ],
                width=130
            )

        except Exception as error:

            st.error(
                f"Video analysis failed: {error}"
            )

        finally:

            if video_path is not None:
                video_path.unlink(missing_ok=True)


# =====================================================
# 16. LABEL INFORMATION
# =====================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    'Binary Classification Labels'
    '</div>',
    unsafe_allow_html=True
)

label1, label2 = st.columns(2)

with label1:

    st.success(
        "**Label 0 — Normal Behaviour**"
    )

    st.caption(
        "Video classified into the normal behaviour class."
    )

with label2:

    st.warning(
        "**Label 1 — Abnormal Behaviour**"
    )

    st.caption(
        "Video classified into the abnormal behaviour class."
    )


# =====================================================
# 17. MODEL ARCHITECTURE INFORMATION
# =====================================================

with st.expander("View Model Architecture"):

    st.markdown("""
    ### CNN Feature Extractor

    - MobileNetV2 pretrained on ImageNet
    - 1280-dimensional feature vectors
    - 16 uniformly sampled frames

    ### Temporal Classifier

    - Linear projection: 1280 → 128
    - ReLU activation
    - Bidirectional LSTM
    - Hidden size: 64
    - One LSTM layer
    - Mean temporal pooling
    - Classifier: 128 → 64 → 2

    ### Inference Configuration

    - CPU execution
    - Saved training normalization
    - Softmax class probabilities
    - Experiment 2 trained checkpoint
    """)


# =====================================================
# 18. FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        CNN-BiLSTM Video Anomaly Detection<br>
        DCSASS Dataset | MobileNetV2 + BiLSTM | CPU Inference
    </div>
    """,
    unsafe_allow_html=True
)
