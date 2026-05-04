# Fake vs Real Face Detector

CNN-based image classifier that predicts whether an uploaded face image is real or AI-generated/fake. The project includes a TensorFlow training script and a Streamlit app for testing images through a simple web interface.

## Features

- Trains a binary classifier on real and fake face images
- Uses TensorFlow/Keras CNN layers for image classification
- Loads image paths and labels from `metadata.csv`
- Saves trained model to `model/fake_real_detector.h5`
- Provides Streamlit UI for uploading an image and viewing prediction confidence

## Dataset

Dataset source:

https://www.kaggle.com/datasets/troykueh/real-vs-fake-faces-stylegan3

Expected project structure:

```text
.
├── Fake faces/
├── Real faces/
├── metadata.csv
├── train_model.py
├── app.py
├── requirements.txt
└── model/
```

`metadata.csv` should contain:

```csv
filepath,label
Fake faces/fake_9650.png,fake
Real faces/real_2041.png,real
```

Current dataset layout uses 20,000 images total:

- 10,000 fake face images
- 10,000 real face images

## Installation

Clone repository:

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate virtual environment:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For model training, install training dependencies instead:

```bash
pip install -r requirements-train.txt
```

## Train Model

Make sure `Fake faces/`, `Real faces/`, and `metadata.csv` are in the project root, then run:

```bash
python train_model.py
```

Training saves the model here:

```text
model/fake_real_detector.h5
```

## Run App

After training, start the Streamlit app:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal and upload a `.jpg`, `.jpeg`, or `.png` face image.

## Deploy on Streamlit Cloud

The deployed app uses NumPy inference through `model/fake_real_detector_weights.npz`, so Streamlit Cloud does not need TensorFlow.

If you retrain the model, export updated NumPy weights before deploying again.

## Model

The model is a simple convolutional neural network:

- 3 convolution + max pooling blocks
- Flatten layer
- Dense layer with dropout
- Sigmoid output for binary classification

Labels:

- `0` = real
- `1` = fake

## Notes

- The app expects `model/fake_real_detector.h5` to exist before launch.
- Training script expects PNG images because it uses `tf.image.decode_png`.
- Large datasets and trained model files may be too large for GitHub. Consider using `.gitignore` or Git LFS for image folders and `.h5` model files.

## Tech Stack

- Python
- TensorFlow/Keras
- Pandas
- scikit-learn
- Streamlit
- Pillow
