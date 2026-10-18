
import streamlit as st
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from streamlit_drawable_canvas import st_canvas
from pathlib import Path


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="MNIST CNN Digit Recognizer",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 MNIST Digit Recognizer")
st.write(
    "Draw a digit from 0 to 9 and let your PyTorch CNN "
    "predict what you wrote."
)


# --------------------------------------------------
# 2. Recreate your trained CNN architecture
# --------------------------------------------------

def create_model():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(16, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),

        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),

        nn.Dropout(0.25),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(256, 10)
    )


# --------------------------------------------------
# 3. Load the checkpoint once
# --------------------------------------------------

@st.cache_resource
def load_model():
    checkpoint_path = Path(__file__).parent / "best_mnist_cnn.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    model = create_model()

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )
    except TypeError:
        # Compatibility with older PyTorch versions
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu"
        )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return model


try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()


# --------------------------------------------------
# 4. Draw a digit
# --------------------------------------------------

st.subheader("Draw your digit")

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0


canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=18,
    stroke_color="#FFFFFF",
    background_color="#000000",
    drawing_mode="freedraw",
    height=280,
    width=280,
    update_streamlit=True,
    return_image_data=True,
    key=f"digit_canvas_{st.session_state.canvas_key}"
)


predict_col, clear_col = st.columns(2)

with predict_col:
    predict_clicked = st.button(
        "Predict digit",
        type="primary",
        use_container_width=True
    )

with clear_col:
    clear_clicked = st.button(
        "Clear canvas",
        use_container_width=True
    )

if clear_clicked:
    st.session_state.canvas_key += 1
    st.rerun()


# --------------------------------------------------
# 5. Preprocess the drawing for MNIST
# --------------------------------------------------

def preprocess_drawing(image_data):
    # Canvas produces RGBA pixels; MNIST uses grayscale.
    image = Image.fromarray(
        image_data.astype(np.uint8)
    ).convert("L")

    # Find the area containing the white digit.
    mask = image.point(lambda pixel: 255 if pixel > 30 else 0)
    bbox = mask.getbbox()

    if bbox is None:
        return None, None

    # Crop around the digit.
    digit = image.crop(bbox)

    # Resize while preserving aspect ratio.
    digit.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Place the digit in the center of a black 28x28 image.
    mnist_image = Image.new("L", (28, 28), color=0)

    x_offset = (28 - digit.width) // 2
    y_offset = (28 - digit.height) // 2

    mnist_image.paste(digit, (x_offset, y_offset))

    # Convert pixel values from 0-255 to 0-1.
    image_array = np.array(
        mnist_image,
        dtype=np.float32
    ) / 255.0

    # Shape: [batch, channel, height, width]
    image_tensor = torch.from_numpy(
        image_array
    ).unsqueeze(0).unsqueeze(0)

    return mnist_image, image_tensor


# --------------------------------------------------
# 6. Run inference
# --------------------------------------------------

if predict_clicked:
    if canvas_result.image_data is None:
        st.warning("Draw a digit on the canvas first.")
    else:
        processed_image, input_tensor = preprocess_drawing(
            canvas_result.image_data
        )

        if input_tensor is None:
            st.warning("The canvas is empty. Please draw a digit.")
        else:
            with torch.inference_mode():
                logits = model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)[0]
                predicted_digit = int(torch.argmax(probabilities).item())

            confidence = float(
                probabilities[predicted_digit].item()
            ) * 100

            st.divider()
            st.subheader("Prediction result")

            result_col, image_col = st.columns(2)

            with result_col:
                st.metric(
                    label="Predicted digit",
                    value=predicted_digit
                )
                st.metric(
                    label="Model confidence",
                    value=f"{confidence:.2f}%"
                )

            with image_col:
                st.image(
                    processed_image,
                    caption="Preprocessed 28 × 28 image",
                    width=140
                )

            st.subheader("Class probabilities")

            probability_data = {
                str(digit): float(probabilities[digit].item())
                for digit in range(10)
            }

            st.bar_chart(probability_data)

            st.caption(
                "Confidence is the model's softmax probability, "
                "not a guarantee that the prediction is correct."
            )
