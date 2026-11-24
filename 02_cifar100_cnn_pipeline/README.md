# 🖼️ CIFAR-100 Deep Image Classification Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sahilgoury7/genai-lab/blob/main/02_cifar100_cnn_pipeline/cifar100_project.ipynb)
[![GitHub](https://img.shields.io/badge/GitHub-genai--lab-181717?style=flat&logo=github&logoColor=white)](https://github.com/sahilgoury7/genai-lab)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Sahil%20Goury-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sahil-goury-269107370/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x%20%2F%20Keras%203-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Completed%20%E2%9C%85-success)](#)

> An end-to-end Deep Learning Computer Vision pipeline on the challenging **CIFAR-100** dataset (100 real-world fine-grained object categories). This project demonstrates the transition from a naive baseline CNN suffering from massive overfitting to an optimized production-grade CNN regularized with **Data Augmentation**, **Batch Normalization**, **Smart Adaptive Callbacks**, and **Center-Crop Real-World Inference**.

---

## 👨‍💻 Author & Connect

* **Developer:** [Sahil Goury](https://www.linkedin.com/in/sahil-goury-269107370/)
* **Repository:** [github.com/sahilgoury7/genai-lab](https://github.com/sahilgoury7/genai-lab)
* **Interactive Notebook:** [Run Live on Google Colab](https://colab.research.google.com/github/sahilgoury7/genai-lab/blob/main/02_cifar100_cnn_pipeline/cifar100_project.ipynb)

---

## 📌 Project Overview & Key Highlights

* **100 Fine-Grained Object Classes:** Classified complex categories with limited training data (only 500 images per class at $32 \times 32$ RGB resolution).
* **Deep 3-Block Architecture:** Designed a modular CNN with $32 \to 64 \to 128$ filters, Batch Normalization, MaxPooling, and a regularized Dense classification head (**Total Params: 644,388**).
* **Diagnosing & Curing Severe Overfitting:**
  - *Baseline Model:* Hit a massive **38% generalization gap** (Train Acc ~79% vs Val Acc ~41%) and loss explosion ($0.65 \to 3.23$).
  - *Regularized Model:* Integrated real-time **Data Augmentation** (`RandomFlip`, `RandomRotation`, `RandomZoom`) and smart callbacks (`EarlyStopping`, `ReduceLROnPlateau`), completely curing overfitting ($\text{Val Acc} > \text{Train Acc}$).
* **32% Test Loss Reduction:** Slashed test loss from **3.16 to 2.1498**.
* **Top-5 Accuracy of ~71%:** Evaluated beyond Top-1 accuracy to reflect true semantic ranking across 100 classes.
* **Production Inference Pipeline:** Implemented an aspect-ratio-preserving **Center Crop Preprocessing** pipeline for high-resolution internet images.

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
│  Flatten (2,048 Clues)                       │ ──▶ 1D Feature Vector
└──────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────┐
│  Dense(256) + Dropout(0.5)                   │ ──▶ 524,544 Parameters
└──────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────┐
│  Dense(100, activation='softmax')            │ ──▶ Probabilities (Sum = 1.0)
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

| Metric | Model 1: Baseline CNN | Model 2: Augmented Pro CNN | Impact / Resolution |
| :--- | :---: | :---: | :--- |
| **Data Augmentation** | None | `RandomFlip`, `RandomRotation(0.1)`, `RandomZoom(0.1)` | Synthesizes novel views every epoch |
| **Smart Callbacks** | None | `EarlyStopping` + `ReduceLROnPlateau` | Epoch 18: LR dropped $0.001 \to 0.0005$; Best weights saved |
| **Training Accuracy** | $79.15\%$ *(Overfit)* | $\mathbf{39.24\%}$ *(Honest learning)* | Model prevented from brute-force memorization |
| **Validation Accuracy**| $41.33\%$ | $\mathbf{42.96\%}$ | **Generalization Restored** ($\text{Val} > \text{Train}$) |
| **Final Test Loss** | $3.1635$ 💥 | $\mathbf{2.1498}$ 📉 | **32% Error Reduction** |
| **Final Test Accuracy**| $42.50\%$ | $\mathbf{43.93\%}$ | Robust performance across all test subsets |
| **Top-5 Accuracy** | ~66% | $\mathbf{71.4\%}$ | High semantic relevance among top predictions |

---

## 🔬 Real-World Case Studies & Engineering Insights

### 1. The "Sea vs Television" Case Study (Context Bias)
* **Ground Truth:** Class 71 $\implies$ `sea` (Blue ocean visible between two vertical tree trunks).
* **Model 1 Prediction:** Class 87 $\implies$ `television` (**84.44% confidence** ❌).
* **Root Cause Analysis:** The baseline model saw dark vertical tree trunks as a TV bezel framing bright blue water as a TV screen.
* **Model 2 Remedy:** With data augmentation, camera angles and boundaries shifted dynamically—dropping the blind overconfidence on television by nearly half (**47.2%**).

### 2. Real-World Internet Image Inference (Center Crop vs Direct Resize)
Testing on an external high-resolution Apple photo downloaded from the web:
* **Direct Resize ($32 \times 32$):** Squashed the aspect ratio and filled 80% of the canvas with floor background. The model predicted `cockroach` ($17.2\%$).
* **Center Crop ($32 \times 32$):** Discarded surrounding background noise and centered the fruit. The false cockroach detection vanished, and `apple` shot straight into the **Top 3 predictions**!

### 3. CIFAR-100 Dataset Vocabulary Nuance
When testing photos of a *Cat* and a *Car*:
* The model predicted `fox` for a cat, and `motorcycle` for a car.
* **Explanation:** Domestic `cat` and consumer `automobile/car` are **not** part of CIFAR-100's 100 classes (CIFAR-100 contains wild carnivores like `fox`, `wolf`, `lion`, and vehicles like `bus`, `pickup_truck`, `motorcycle`). The CNN extracted the right features (whiskers, fur, wheels) and mapped them to the nearest semantic neighbors in its vocabulary!

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
