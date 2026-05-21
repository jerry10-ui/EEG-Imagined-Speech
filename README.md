# EEG Imagined Speech Recognition

An end-to-end deep learning system for decoding imagined speech from EEG signals using CNNs and a Streamlit-based interactive web application.

## Overview

This project takes raw EEG '.edf' data, preprocesses it into segments, trains a convolutional neural network (CNN) for classification, and allows interactive predictions through a Streamlit web interface.

> Note:
> The dataset files (`.edf`, `.npy`, `.h5`, etc.) are not included in this repository due to size limitations.  
> You can recreate them locally using the provided scripts and dataset links.

---

## Project Structure

| File | Purpose |
|------|----------|
| `loader.py` | Loads and converts `.edf` EEG files to array format. |
| `segmenter.py` | Segments EEG data into frames `(39 × 256)`. |
| `clean.py` | Cleans and normalizes EEG signals. |
| `verify.py` | Verifies NaNs, array shapes, and sample counts. |
| `train_model.py` | Builds, trains, and saves the CNN model. |
| `plot_curves.py` | Plots training accuracy/loss and confusion matrix. |
| `app.py` | Streamlit web app for uploading `.npy` and predicting. |
| `.gitignore` | Ignored files and folders. |
| `requirements.txt` | Python dependencies list. |
| `README.md` | Complete documentation. |

---

### Dataset Used — *Imagined Speech EEG Dataset*
https://www.kaggle.com/datasets/ignazio/kumars-eeg-imagined-speech

* **Dataset Name:** *Kumar's EEG Imagined speech*

* **Source:** Publicly available dataset (often used in EEG signal classification research).

* **Structure of the dataset:**

  ```
  /Imagined_speech_EEG_edf/
  ├── Char/
  ├── Digit/
  └── Image/
  ```

* **Format:**
  Each file is in `.edf` (European Data Format) containing EEG signals recorded from multiple electrodes.
  You converted these `.edf` files into `.npy` arrays using `loader.py`.

* **Content:**
  EEG signals were recorded while subjects imagined speaking **10 different words**:

  ```
  Apple, Car, Dog, Gold, Mobile, Rose, Scooter, Tiger, Wallet, Watch
  ```

* **Sampling:**
  Each sample was segmented into a **39×256** frame representing EEG channels × time points.

* **Preprocessing done:**

  1. Data loading from `.edf` files
  2. Segmentation into uniform time frames
  3. Cleaning (removal of NaN/inf values)
  4. Normalization (mean = 0, std = 1)
  5. Verification of shape and integrity

* **Purpose:**
  To train a CNN model capable of decoding imagined speech patterns directly from EEG data.

---

### 1. Setup Instructions

```bash
# Clone this repository
git clone https://github.com/jerry10-ui/ML-Imagined-Speech.git
cd EEG-Imagined-Speech

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate     # On Windows
# or
source .venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
````

---

### 2. Requirements

```
numpy
mne
scikit-learn
tensorflow>=2.12
matplotlib
seaborn
streamlit
gTTS
pyttsx3
h5py
```

---

### 3. Base Directory and File Paths

Each script begins with a "BASE_DIR" variable to define where your files are located.

Before running any script, make sure you’ve updated the paths:

```python
# Example in each script
BASE_DIR = r"C:\Coding\ML Imagined Speech"

MODEL_PATH = os.path.join(BASE_DIR, "eeg_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "label_classes.npy")
X_SEGMENTS_PATH = os.path.join(BASE_DIR, "X_segments.npy")
Y_SEGMENTS_PATH = os.path.join(BASE_DIR, "y_segments.npy")
```

> Note: Keep all files inside one main folder (like the path above) to simplify this setup.

---

### 4. Data Preprocessing Pipeline

### Step 1 — Load EEG Files

```bash
python loader.py
```

Loads `.edf` EEG recordings and converts them into array format.

### Step 2 — Segment EEG

```bash
python segmenter.py
```

Creates `(39 × 256)` EEG data segments and saves:

* `X_segments.npy`
* `y_segments.npy`

### Step 3 — Clean & Normalize

```bash
python clean.py
```

Removes NaN values and normalizes EEG signals for consistent training.

### Step 4 — Verify Data

```bash
python verify.py
```

Displays shape, NaN count, and verifies preprocessing success.

---

### 5. Model Training

```bash
python train_model.py
```

Performs:

* Label encoding
* Train-test split
* CNN model training
* Model saving

The trained model and labels are saved as:

```
eeg_model.h5
label_classes.npy
history.npy
```

### Dataset Splitting Logic

```python
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)
```

→ 80% training, 10% validation, 10% testing

---

### 6. Model Evaluation

```bash
python plot_curves.py
```

Outputs:

* `training_curves.png` — Accuracy/Loss curves
* `confusion_matrix.png` — Heatmap
* `classification_report.txt` — Precision, Recall, F1-Score

---

### 7. Interactive Inference

### Streamlit App

```bash
streamlit run app.py
```

Features:

* Upload `.npy` EEG files
* Predicts class probabilities
* Text-to-speech output of top prediction

If audio doesn’t play automatically, switch between **`gTTS`** and **`pyttsx3`** in the code.

---

### 8. .gitignore Summary

All large or temporary files are ignored:

```
*.npy
*.h5
*.mp3
*.wav
__pycache__/
.venv/
.streamlit/
```

---

## Author

Developed by Abhay Garg
(Pursuing B.Tech CSE – AI/ML)

---
