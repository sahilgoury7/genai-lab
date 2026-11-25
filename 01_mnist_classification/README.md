# 🔢 MNIST Handwritten Digit Classification (ANN)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sahilgoury7/genai-lab/blob/main/01_mnist_classification/mnist_classification.ipynb)
[![GitHub](https://img.shields.io/badge/GitHub-genai--lab-181717?style=flat&logo=github&logoColor=white)](https://github.com/sahilgoury7/genai-lab)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Sahil%20Goury-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sahil-goury-269107370/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x%20%2F%20Keras-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Completed%20%E2%9C%85-success)](#)

> An end-to-end Computer Vision foundational project classifying handwritten digits (0–9) using a Multi-Layer **Artificial Neural Network (ANN)** built with **TensorFlow / Keras**. This project explores image flattening, pixel normalization, parameter mathematics, and multi-class probability estimation.

---

## 👨‍💻 Author & Connect

* **Developer:** [Sahil Goury](https://www.linkedin.com/in/sahil-goury-269107370/)
* **Repository:** [github.com/sahilgoury7/genai-lab](https://github.com/sahilgoury7/genai-lab)
* **Interactive Notebook:** [Run Live on Google Colab](https://colab.research.google.com/github/sahilgoury7/genai-lab/blob/main/01_mnist_classification/mnist_classification.ipynb)

---

## 📌 Project Overview & Highlights

* **Dataset:** MNIST dataset consisting of **70,000 grayscale images** ($28 \times 28$ resolution) of handwritten digits from 0 to 9.
  * **Training Split:** 48,000 images (from 60,000 training pool with 20% validation split).
  * **Validation Split:** 12,000 images for real-time generalization tracking.
  * **Testing Split:** 10,000 unseen images for unbiased test evaluation.
* **Lightweight ANN Architecture:** Sequential network (`Flatten` $\to$ `Dense(128, ReLU)` $\to$ `Dense(10, Softmax)`).
* **Parameter Mathematics:** Total trainable parameters: **101,770** weights and biases.
* **Accuracy:** Achieved **97.27% Test Accuracy** (Test Loss: **0.1022**) on unseen test digits within 10 training epochs.
* **Visual Inspection:** Visualizes sample predictions and plots training vs. validation loss/accuracy curves.

---

## 🏗️ Architecture Pipeline & Parameter Mathematics

```
[28x28 Grayscale Image] (Pixel values: 0 to 255)
         │
         ▼  (Step 1: Normalization)
   X / 255.0  ──▶ (Pixel values: 0.0 to 1.0)
         │
         ▼  (Step 2: Flatten Layer)
   Flatten(input_shape=(28, 28)) ──▶ [784 1D Feature Vector]
         │
         ▼  (Step 3: Hidden Dense Layer)
   Dense(128, activation='relu') ──▶ (784 × 128) + 128 = 100,480 Parameters
         │
         ▼  (Step 4: Output Dense Layer)
   Dense(10, activation='softmax') ──▶ (128 × 10) + 10 = 1,290 Parameters
         │
         ▼  (Step 5: Prediction)
   np.argmax(y_prob, axis=1) ──▶ Final Predicted Class (0 to 9)
```

### 📐 Layer Parameter Breakdown

| Layer | Type | Output Shape | Formula | Trainable Params |
| :--- | :--- | :---: | :--- | :---: |
| `flatten` | Flatten | `(None, 784)` | Non-parametric reshape ($28 \times 28$) | **0** |
| `dense_hidden` | Dense (ReLU) | `(None, 128)` | $(\text{Inputs} \times \text{Nodes}) + \text{Biases} = (784 \times 128) + 128$ | **100,480** |
| `dense_output` | Dense (Softmax) | `(None, 10)` | $(\text{Inputs} \times \text{Nodes}) + \text{Biases} = (128 \times 10) + 10$ | **1,290** |
| **Total** | | | **Trainable: 101,770 \| Non-trainable: 0** | **101,770** |

---

## 📊 Experimental Results

*(Exact figures from the executed notebook)*

| Metric | Training Set (Epoch 10) | Validation Set (Epoch 10) | Unseen Test Set (10k images) |
| :--- | :---: | :---: | :---: |
| **Loss** | **0.0168** | **0.1113** | **0.1022** |
| **Accuracy** | **99.48%** | **97.13%** | **97.27%** |

> 💡 **Training Trajectory Insight:** The lowest validation loss was reached at **Epoch 6 (`val_loss: 0.0844`, `val_accuracy: 97.49%`)**. Past Epoch 6, training loss continued to drop to $0.0168$ while validation loss slightly increased to $0.1113$ — demonstrating the textbook onset of mild overfitting, which can be mitigated with Dropout or EarlyStopping.

---

## 🔬 Core Deep Learning Concepts Explored

### 1. Why Pixel Normalization ($X / 255.0$)?
Raw pixels range from $0$ to $255$. When unnormalized values enter matrix multiplications, activations explode and gradients oscillate wildly during Backpropagation. Dividing by $255.0$ maps all features uniformly into the range $[0.0, 1.0]$, ensuring smooth, fast, and numerically stable Gradient Descent optimization.

### 2. ReLU vs Softmax Activations
* **ReLU ($\max(0, x)$):** Used in the hidden layer to introduce non-linearity while preventing vanishing gradients for positive inputs.
* **Softmax ($\frac{e^{z_i}}{\sum e^{z_j}}$):** Used in the final classification layer to transform raw logits into a normalized probability distribution across the 10 digit classes (where $\sum P_i = 1.0$).

### 3. Sparse Categorical Crossentropy vs Categorical Crossentropy
* We selected **`sparse_categorical_crossentropy`** because our labels are integer targets ($y \in \{0, 1, 2, \dots, 9\}$).
* If labels were one-hot encoded vectors (e.g., $[0, 0, 1, 0, \dots, 0]$), standard `categorical_crossentropy` would be required. `sparse_categorical_crossentropy` saves memory by avoiding one-hot vector allocations across thousands of samples.

---

## 💡 The Big Question: Why Move from ANN to CNN?

While this ANN achieves an impressive **97.27% test accuracy** on MNIST, it highlights the fundamental limitation of standard Multi-Layer Perceptrons for Computer Vision:

| Feature | Artificial Neural Network (ANN) | Convolutional Neural Network (CNN) |
| :--- | :--- | :--- |
| **Input Representation** | **1D Flattened Vector (784)** | **2D Spatial Grid ($28 \times 28 \times 1$)** |
| **Spatial Awareness** | ❌ **Lost** (Neighboring pixels separated) | ✅ **Preserved** ($3\times3$ filters scan local patches) |
| **Translation Invariance** | ❌ Poor (Shifted digit looks completely new) | ✅ High (Detects strokes anywhere in the frame) |
| **Parameter Scaling** | Explodes on larger/RGB images | Parameter sharing keeps weights small |

> ⏭️ **Next Project in Journey:** [CIFAR-100 Deep Image Classification Pipeline](../02_cifar100_cnn_pipeline) — solving spatial vision and complex 100-class RGB datasets with modern CNN blocks, Data Augmentation, and Batch Normalization.

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/sahilgoury7/genai-lab.git
cd genai-lab/01_mnist_classification
```

### 2. Install Dependencies
```bash
pip install tensorflow numpy matplotlib jupyter
```

### 3. Launch the Notebook
```bash
jupyter notebook mnist_classification.ipynb
```
