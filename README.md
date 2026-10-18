# 🔢 MNIST CNN Digit Recognizer

A handwritten digit recognition application built using **PyTorch** and **Streamlit**. Draw a digit from 0 to 9 on the interactive canvas, and the trained Convolutional Neural Network (CNN) predicts the digit along with its class probabilities.

## 🚀 Live Demo

**Coming soon!** The live demo will be added after deployment.

## ✨ Features

- ✍️ Interactive canvas for drawing handwritten digits.
- 🧠 Custom CNN built and trained using PyTorch.
- 🔟 Classification of digits from 0 to 9.
- 📊 Prediction confidence and class probability visualization.
- 🖼️ Displays the preprocessed 28 × 28 image supplied to the model.
- ⚡ Efficient inference using `torch.inference_mode()`.
- 🔄 Clear the canvas and test multiple digits.

## 🛠️ Tech Stack

- **Python** — Core programming language
- **PyTorch** — CNN architecture and model inference
- **Streamlit** — Interactive web interface
- **NumPy** — Numerical operations and tensor preparation
- **Pillow (PIL)** — Image processing
- **streamlit-drawable-canvas** — Interactive drawing canvas

## 🧠 Model Architecture

The CNN consists of:

- Four convolutional layers with increasing channel dimensions: 16, 32, 64, and 128.
- Batch normalization and ReLU activation.
- Max pooling for spatial downsampling.
- Adaptive average pooling.
- Fully connected layers with dropout.
- A final output layer containing 10 logits, one for each digit class.

The model accepts grayscale images with the shape **`[N, 1, 28, 28]`** and produces 10 output logits per image. Softmax converts these logits into class probabilities for display.

## 📈 Model Evaluation

The model achieved approximately **99.06% accuracy** on the MNIST test set in the recorded evaluation.

- Test samples: 10,000
- Correct predictions: 9,906
- Incorrect predictions: 94

Performance may vary depending on the checkpoint used.

## 📂 Project Structure

```text
mnist-cnn-digit-recognizer/
├── app.py                  # Streamlit application
├── best_mnist_cnn.pth      # Trained PyTorch checkpoint
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── .gitignore              # Git exclusions
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/mnist-cnn-digit-recognizer.git
cd mnist-cnn-digit-recognizer
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Open the local URL shown in your terminal, usually `http://localhost:8501`.

## 🎮 How to Use

1. Open the application.
2. Draw a digit between 0 and 9 using the canvas.
3. Click **Predict digit**.
4. View the predicted digit and its softmax probability.
5. Inspect the preprocessed 28 × 28 image and probability distribution.
6. Click **Clear canvas** to try another digit.

## 🔍 Image Preprocessing

The drawing is prepared for the model through the following steps:

1. Convert the canvas image to grayscale.
2. Locate and crop the drawn digit.
3. Resize it while preserving its aspect ratio.
4. Center it on a black 28 × 28 canvas.
5. Normalize pixel intensities to the range `[0, 1]`.
6. Convert the image into a PyTorch tensor with the expected input dimensions.

## ⚠️ Limitations

- Predictions depend on how clearly the digit is drawn.
- Handwritten styles may differ from the MNIST training distribution.
- Softmax probability is not a guarantee that a prediction is correct.
- The application is intended for educational and demonstration purposes.

## 🔮 Future Improvements

- Improve digit centering using center-of-mass alignment.
- Display top-k predictions.
- Add a gallery of incorrect predictions from model evaluation.
- Compare CNN performance with other architectures.
- Add automated tests and model evaluation metrics.
- Deploy the application publicly.

## 👨‍💻 Author

**Rushabh Borkar**

Computer Science Engineering — AI & Machine Learning

- GitHub: https://github.com/rushabh-borkar

