
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Gun Image Classification With AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(34, 197, 94, 0.10),
                transparent 35%
            ),
            radial-gradient(
                circle at 90% 90%,
                rgba(16, 185, 129, 0.08),
                transparent 35%
            ),
            #07110d;
        color: #ecfdf5;
    }

    /* Main content */
    .block-container {
        max-width: 1100px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .hero {
        text-align: center;
        padding: 20px 10px 30px 10px;
    }

    .hero-badge {
        display: inline-block;
        padding: 7px 16px;
        border-radius: 30px;
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(74, 222, 128, 0.25);
        color: #86efac;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 14px;
    }

    .hero h1 {
        font-size: 48px;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(
            90deg,
            #f0fdf4,
            #86efac
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero p {
        color: #a7b8af;
        font-size: 17px;
        margin-top: 10px;
    }

    /* Cards */
    .glass-card {
        background: rgba(15, 35, 27, 0.72);
        border: 1px solid rgba(134, 239, 172, 0.12);
        border-radius: 20px;
        padding: 24px;
        box-shadow:
            0 10px 40px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(12px);
    }

    /* Section title */
    .section-title {
        color: #d1fae5;
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 15px;
    }

    /* Prediction result */
    .prediction-card {
        background: linear-gradient(
            135deg,
            rgba(22, 101, 52, 0.35),
            rgba(6, 78, 59, 0.30)
        );
        border: 1px solid rgba(74, 222, 128, 0.25);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin-top: 20px;
    }

    .prediction-label {
        color: #86efac;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 700;
    }

    .prediction-name {
        color: #f0fdf4;
        font-size: 32px;
        font-weight: 800;
        margin: 8px 0;
    }

    .confidence {
        color: #a7f3d0;
        font-size: 18px;
        font-weight: 600;
    }

    /* Info cards */
    .info-card {
        background: rgba(15, 35, 27, 0.65);
        border: 1px solid rgba(134, 239, 172, 0.10);
        border-radius: 16px;
        padding: 18px;
        text-align: center;
        height: 100%;
    }

    .info-number {
        font-size: 25px;
        font-weight: 800;
        color: #86efac;
    }

    .info-text {
        font-size: 13px;
        color: #9caea5;
        margin-top: 5px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64756d;
        font-size: 12px;
        margin-top: 45px;
        padding-top: 20px;
        border-top: 1px solid rgba(134, 239, 172, 0.08);
    }

    /* Streamlit uploader */
    [data-testid="stFileUploader"] {
        background: rgba(15, 35, 27, 0.55);
        border: 1px dashed rgba(134, 239, 172, 0.30);
        border-radius: 16px;
        padding: 10px;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid rgba(134, 239, 172, 0.25);
        background: linear-gradient(
            90deg,
            #166534,
            #047857
        );
        color: white;
        font-weight: 700;
        padding: 12px;
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #86efac;
        transform: translateY(-1px);
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = (
    "models/best_mobilenetv2_augmented_v3.keras"
)

CLASS_NAMES = [
    "Assault_Rifles",
    "Handguns",
    "Machine_Guns",
    "Revolvers",
    "Shotguns",
    "SMG"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_classification_model():

    if not os.path.exists(MODEL_PATH):
        return None

    return tf.keras.models.load_model(
        MODEL_PATH,
        safe_mode=False
    )


model = load_classification_model()


# ============================================================
# HEADER
# ============================================================

st.caption("FIREARMS IMAGE CLASSIFICATION")

st.title("Gun Classification With AI")

st.write(
    "Upload an image and let our MobileNetV2 model "
    "identify the most likely gun category."
)


# ============================================================
# MODEL STATUS
# ============================================================

if model is None:

    st.error(
        "Model could not be loaded. "
        "Please check the Google Drive model path."
    )

    st.stop()

else:

    st.success("✓ MobileNetV2 V3 model loaded successfully.")


# ============================================================
# INFORMATION CARDS
# ============================================================

col1, col2, col3 = st.columns(3)

card_style = """
    background: rgba(15, 35, 27, 0.72);
    border: 1px solid rgba(134, 239, 172, 0.14);
    border-radius: 18px;
    padding: 20px;
    text-align: center;
    min-height: 105px;
"""

with col1:
    st.markdown(
        f"""
        <div style="{card_style}">
            <div style="
                font-size:26px;
                font-weight:800;
                color:#86efac;
            ">
                6
            </div>
            <div style="
                font-size:13px;
                color:#a7b8af;
                margin-top:5px;
            ">
                Gun Categories
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div style="{card_style}">
            <div style="
                font-size:26px;
                font-weight:800;
                color:#86efac;
            ">
                V3
            </div>
            <div style="
                font-size:13px;
                color:#a7b8af;
                margin-top:5px;
            ">
                MobileNetV2 Model
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div style="{card_style}">
            <div style="
                font-size:26px;
                font-weight:800;
                color:#86efac;
            ">
                AI
            </div>
            <div style="
                font-size:13px;
                color:#a7b8af;
                margin-top:5px;
            ">
                Image Classification
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader("📤 Upload an Image")

st.write(
    "Upload a JPG, JPEG, or PNG image to classify the gun type."
)

uploaded_file = st.file_uploader(
    "Choose a gun image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# IMAGE PREVIEW + PREDICTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.write("")

    left, right = st.columns([1, 1])

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    with left:

        st.markdown(
            '<div class="section-title">🖼️ Image Preview</div>',
            unsafe_allow_html=True
        )

        st.image(
            image,
            use_container_width=True
        )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with right:

        st.markdown(
            '<div class="section-title">🔍 Classification</div>',
            unsafe_allow_html=True
        )

        st.write("")

        predict_button = st.button(
            "🚀 Analyze Image"
        )

        if predict_button:

            with st.spinner("Analyzing image..."):

                # Convert image to array
                img_array = np.array(image)

                # Resize to model input size
                img_array = tf.image.resize(
                    img_array,
                    (160, 160)
                )

                # Normalize exactly like training
                img_array = tf.cast(
                    img_array,
                    tf.float32
                ) / 255.0

                # Add batch dimension
                img_array = tf.expand_dims(
                    img_array,
                    axis=0
                )

                # Prediction
                prediction = model.predict(
                    img_array,
                    verbose=0
                )[0]

                predicted_index = np.argmax(prediction)

                predicted_class = CLASS_NAMES[
                    predicted_index
                ]

                confidence = (
                    prediction[predicted_index] * 100
                )


            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

                        # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.success("Prediction completed successfully")

            st.markdown("### 🎯 Predicted Gun Type")

            st.markdown(
                f"## **{predicted_class.replace('_', ' ')}**"
            )

            st.markdown(
                f"### Confidence: **{confidence:.2f}%**"
            )

            st.progress(
                float(confidence / 100)
            )

            st.write("")


            # ------------------------------------------------
            # PROBABILITY DISTRIBUTION
            # ------------------------------------------------

            st.write("")

            st.markdown(
                '<div class="section-title">📊 Prediction Probabilities</div>',
                unsafe_allow_html=True
            )

            probabilities = prediction * 100

            for class_name, probability in zip(
                CLASS_NAMES,
                probabilities
            ):

                st.write(
                    f"**{class_name.replace('_', ' ')}**"
                )

                st.progress(
                    float(probability / 100)
                )

                st.caption(
                    f"{probability:.2f}%"
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    Gun Classification AI • MobileNetV2 V3
    <div>AI Laboratory Project</div>
</div>
""", unsafe_allow_html=True)
