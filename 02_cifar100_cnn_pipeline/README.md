# 🖼️ CIFAR-100 Deep Image Classification Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sahilgoury7/genai-lab/blob/main/02_cifar100_cnn_pipeline/cifar100_project.ipynb)
[![GitHub](https://img.shields.io/badge/GitHub-genai--lab-181717?style=flat&logo=github&logoColor=white)](https://github.com/sahilgoury7/genai-lab)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Sahil%20Goury-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sahil-goury-269107370/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x%20%2F%20Keras%203-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Completed%20%E2%9C%85-success)](#)

> An end-to-end Deep Learning Computer Vision pipeline on the challenging **CIFAR-100** dataset (100 real-world fine-grained object categories). This project demonstrates the transition from a baseline CNN to an optimized production-grade CNN regularized with **Data Augmentation**, **Batch Normalization**, **Smart Adaptive Callbacks**, and **Center-Crop Real-World Inference**.

---

## 👨‍💻 Author & Connect

* **Developer:** [Sahil Goury](https://www.linkedin.com/in/sahil-goury-269107370/)
* **Repository:** [github.com/sahilgoury7/genai-lab](https://github.com/sahilgoury7/genai-lab)
* **Interactive Notebook:** [Run Live on Google Colab](https://colab.research.google.com/github/sahilgoury7/genai-lab/blob/main/02_cifar100_cnn_pipeline/cifar100_project.ipynb)

---

## 📌 Project Overview & Key Highlights

* **100 Fine-Grained Object Classes:** 50,000 training images and 10,000 test images ($32 \times 32$ RGB resolution) across 100 classes (only 500 images per class).
* **Deep 3-Block Architecture:** Designed a modular CNN with $32 \to 64 \to 128$ filters, Batch Normalization, MaxPooling, and a regularized Dense classification head (**Total Params: 644,388**).
* **Regularization & Callbacks:**
  - Integrated real-time **Data Augmentation** (`RandomFlip`, `RandomRotation(0.1)`, `RandomZoom(0.1)`).
  - Used `BatchNormalization` and `Dropout(0.5)` to stabilize gradients.
  - Applied `EarlyStopping` and `ReduceLROnPlateau` (learning rate dropped from $0.001 \to 0.0005$ at Epoch 18; best weights restored at Epoch 24).
* **Performance Gain:** Test loss reduced from **2.3337 to 2.1498**, with final Test Accuracy reaching **43.93%** (Model 1: 40.14%).
* **Top-5 Accuracy:** Achieved **74.63% Top-5 Accuracy** on 10,000 test images.
* **Production Inference Pipeline:** Implemented aspect-ratio-preserving **Center Crop Preprocessing** for high-resolution internet images.

---

## 🏗️ Model Architecture & Parameter Mathematics

```
Input: (32 x 32 x 3) RGB Image
   │
   ▼
┌──────────────────────────────────────────────┐
│  Conv Block 1: Conv2D(32, 3x3) + BN + Pool   │ ──▶ Output: (16 x 16 x 32)
└──────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────┐
│  Conv Block 2: Conv2D(64, 3x3) + BN + Pool   │ ──▶ Output: (8 x 8 x 64)
└──────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────┐
│  Conv Block 3: Conv2D(128, 3x3) + BN + Pool  │ ──▶ Output: (4 x 4 x 128)
└──────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────┐
│  Flatten (2,048 Features)                    │ ──▶ 1D Feature Vector
└──────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────┐
│  Dense(256, ReLU) + Dropout(0.5)             │ ──▶ 524,544 Parameters
└──────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────┐
│  Dense(100, Softmax)                         │ ──▶ 100 Class Probabilities
└──────────────────────────────────────────────┘
```

### 📐 Layer Parameter Breakdown

| Layer | Type | Output Shape | Param # | Calculation Formula |
| :--- | :--- | :---: | :---: | :--- |
| `conv_block1` | Conv2D (32, 3x3) | `(None, 32, 32, 32)` | **896** | $(3 \times 3 \times 3 + 1) \times 32$ |
| `bn_block1` | BatchNormalization | `(None, 32, 32, 32)` | **128** | $4 \times 32$ ($\gamma, \beta, \mu, \sigma^2$) |
| `pool_block1` | MaxPooling2D (2,2)| `(None, 16, 16, 32)` | **0** | Non-parametric downsampling |
| `conv_block2` | Conv2D (64, 3x3) | `(None, 16, 16, 64)` | **18,496** | $(3 \times 3 \times 32 + 1) \times 64$ |
| `bn_block2` | BatchNormalization | `(None, 16, 16, 64)` | **256** | $4 \times 64$ |
| `pool_block2` | MaxPooling2D (2,2)| `(None, 8, 8, 64)` | **0** | Non-parametric downsampling |
| `conv_block3` | Conv2D (128, 3x3)| `(None, 8, 8, 128)` | **73,856** | $(3 \times 3 \times 64 + 1) \times 128$ |
| `bn_block3` | BatchNormalization | `(None, 8, 8, 128)` | **512** | $4 \times 128$ |
| `pool_block3` | MaxPooling2D (2,2)| `(None, 4, 4, 128)` | **0** | Non-parametric downsampling |
| `flatten` | Flatten | `(None, 2048)` | **0** | $4 \times 4 \times 128 = 2048$ |
| `dense_brain` | Dense (256, ReLU) | `(None, 256)` | **524,544** | $(2048 \times 256) + 256$ |
| `dropout` | Dropout (0.5) | `(None, 256)` | **0** | Randomly zeroes 50% activations |
| `output` | Dense (100, Softmax)| `(None, 100)` | **25,700** | $(256 \times 100) + 100$ |
| **Total** | | | **644,388** | **Trainable: 643,940 \| Non-trainable: 448** |

---

## 🥊 Model 1 (Baseline) vs Model 2 (Augmented Pro) Benchmark

*(Exact figures from the executed notebook)*

| Metric | Model 1: Baseline CNN (10 Epochs) | Model 2: Augmented Pro (25 Epochs + Callbacks) | Key Difference |
| :--- | :---: | :---: | :--- |
| **Data Augmentation** | None | `RandomFlip`, `RandomRotation(0.1)`, `RandomZoom(0.1)` | Prevents memorization of exact pixel locations |
| **Learning Rate Schedule**| Fixed ($0.001$) | Adaptive (`ReduceLROnPlateau`: $0.001 \to 0.0005$ at Epoch 18) | Fine-tunes loss surface convergence |
| **Weight Checkpointing** | None (last epoch) | `EarlyStopping` (Best weights restored at Epoch 24) | Avoids degradation from late epochs |
| **Training Accuracy** | $40.80\%$ | $39.24\%$ | Regularization prevents overconfident memorization |
| **Validation Accuracy**| $40.34\%$ | $\mathbf{42.96\%}$ (Epoch 24) | Improved generalization on unseen validation data |
| **Final Test Loss** | $2.3337$ | $\mathbf{2.1498}$ | **Test loss reduced by 0.1839** |
| **Final Test Accuracy**| $40.14\%$ | $\mathbf{43.93\%}$ | **+3.79% absolute gain** |
| **Top-5 Accuracy** | — | $\mathbf{74.63\%}$ | Almost 3 out of 4 times, true class is in Top-5 |

---

## 🔬 Real-World Case Studies from Notebook

### 1. The "Sea vs Television" Case Study (Context Bias)
* **Image Tested:** Test image #10.
* **Ground Truth:** Class 71 $\implies$ `sea` (Blue water between vertical dark tree trunks).
* **Model 1 Output:** Predicted Class 87 $\implies$ `television` with **73.70% confidence** ❌ (trunks mistaken for a television frame).
* **Model 2 Output:** With augmentation breaking rigid edge alignments, blind confidence on `television` dropped to **47.21%**, and natural background features began entering the candidate distribution.

### 2. Real-World Internet Image Inference (Center Crop vs Direct Resize)
Testing on an external high-resolution Apple photo downloaded from the web:
* **Direct Resize ($32 \times 32$):** Distorted aspect ratio and surrounded the fruit with large background floor area $\implies$ predicted `cockroach` ($17.2\%$).
* **Center Crop ($32 \times 32$):** Cropped the central region, preserving fruit geometry $\implies$ false cockroach prediction eliminated, and `apple` surged into the **Top 3 predictions**.

### 3. CIFAR-100 Dataset Vocabulary Nuance
When testing photos of a *Cat* and a *Car*:
* The model predicted `fox` for a cat, and `motorcycle` for a car.
* **Explanation:** Domestic `cat` and consumer `car/automobile` do not exist in CIFAR-100 classes (which has wild carnivores like `fox`, `wolf`, `lion` and vehicles like `bus`, `pickup_truck`, `motorcycle`). The CNN extracted the correct morphological features (fur/whiskers, wheels/chassis) and mapped them to the closest semantic classes in its 100-class vocabulary.

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/sahilgoury7/genai-lab.git
cd genai-lab/02_cifar100_cnn_pipeline
```

### 2. Install Dependencies
```bash
pip install tensorflow numpy matplotlib pillow scikit-learn jupyter
```

### 3. Launch the Notebook
```bash
jupyter notebook cifar100_project.ipynb
```

---

## 🗺️ Project Navigation

* ⬅️ **Previous Project:** [01_mnist_classification](../01_mnist_classification) — Handwritten Digit Recognition using Feedforward ANN.
