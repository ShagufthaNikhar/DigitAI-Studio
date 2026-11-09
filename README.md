# DigitAI Studio — NNDL Mini Project

A multi-feature handwritten digit recognition application built with TensorFlow/Keras and Streamlit.

## Features
1. **Draw & Predict** — draw a digit with a mouse/touchscreen or upload an image.
2. **Beat the AI** — guess the label of a handwritten MNIST sample, then compare your guess with the CNN prediction.
3. **Analytics Lab** — view actual model metrics, training curves, confusion matrix, and classification report.
4. **CNN Explorer** — inspect the CNN architecture and compare its held-out test accuracy with a basic MLP.
5. **Prediction probabilities** — view the model's score for every digit from 0 to 9.

## Folder structure
```text
DigitAI_Studio_NNDL/
├── app.py
├── train.py
├── predict.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
├── models/
│   ├── mnist_cnn.keras       # created by train.py
│   └── mnist_mlp.keras       # created by train.py
└── reports/
    ├── metrics.json          # created by train.py
    ├── training_curves.png   # created by train.py
    ├── confusion_matrix.png  # created by train.py
    └── classification_report.txt
```

## Setup on Windows (VS Code terminal)
Use Python 3.10 or 3.11, 64-bit.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
streamlit run app.py
```

If PowerShell blocks environment activation:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

First training run downloads MNIST and trains two models. This may take several minutes depending on your computer. Internet is required for the initial dataset download.

## Use
- **Draw & Predict:** draw one large digit in the canvas, then click Predict. You can also upload a photo or image.
- **Beat the AI:** choose your guess before revealing the label. The app compares your guess and the model's prediction with the actual MNIST label.
- **Analytics Lab:** shows saved metrics from your own training run.
- **CNN Explorer:** explains each CNN layer and compares actual test-set results with an MLP.

## Important notes
- The canvas feature uses `streamlit-drawable-canvas`; it requires a browser that supports the component.
- For uploaded photos, use a close, clear image containing a single dark digit on a light background. Image preprocessing uses a simple threshold/connected-component heuristic, so clutter or faint writing can reduce accuracy.
- A model's softmax score is a confidence-like score, not a guarantee that the prediction is correct.
- Do not claim a fixed accuracy in your report. Use the actual `reports/metrics.json` and `reports/classification_report.txt` generated on your computer.

## Syllabus alignment
- **Module 1:** ANN foundations, model training, TensorFlow/Keras.
- **Module 2:** Adam optimizer, dropout regularization, early stopping.
- **Module 3:** CNN, convolution and pooling, image classification.
- **Extension:** MLP baseline comparison, confusion matrix and evaluation metrics.
