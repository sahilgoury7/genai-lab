# 🚀 GenAI & Deep Learning Engineering Lab

[![GitHub](https://img.shields.io/badge/GitHub-sahilgoury7%2Fgenai--lab-181717?style=flat&logo=github&logoColor=white)](https://github.com/sahilgoury7/genai-lab)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Sahil%20Goury-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sahil-goury-269107370/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x%20%2F%20Keras%203-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Status](https://img.shields.io/badge/Status-Actively%20Maintained%20%E2%9C%85-success)](#)

> **Curated Hands-on Engineering Portfolio by Sahil Goury**  
> A structured journey building, training, evaluating, and documenting real-world AI systems: from Foundational Deep Learning (ANN, CNN, RNN) to Modern Generative AI, Fine-Tuning, RAG, and Autonomous Agentic AI.

---

## 🗺️ Master Project Roadmap & Directory

| # | Project Name | Domain & Architecture | Key Results / Highlights | Quick Links | Status |
| :-: | :--- | :--- | :--- | :--- | :-: |
| **01** | [**MNIST Classification**](./01_mnist_classification) | **Computer Vision (ANN)**<br>3-Layer Sequential (`Flatten` $\to$ `Dense(128)` $\to$ `Dense(10)`) | **97.27% Test Accuracy**<br>Analyzed pixel normalization, parameter math & ANN limits on spatial data. | [📓 Notebook](./01_mnist_classification/mnist_classification.ipynb)<br>[📖 README](./01_mnist_classification/README.md) | ✅ Completed |
| **02** | [**CIFAR-100 CNN Pipeline**](./02_cifar100_cnn_pipeline) | **Computer Vision (CNN)**<br>3-Block Modular CNN ($32 \to 64 \to 128$) + BatchNorm + Augmentation | **43.93% Top-1 / 74.63% Top-5**<br>Baseline vs Regularized CNN; test loss reduced from 2.3337 to 2.1498; 74.63% Top-5 accuracy; Center-Crop inference. | [📓 Notebook](./02_cifar100_cnn_pipeline/cifar100_project.ipynb)<br>[📖 README](./02_cifar100_cnn_pipeline/README.md) | ✅ Completed |
| **03** | **Time-Series Stock Predictor** | **Recurrent Networks (RNN / LSTM)**<br>SimpleRNN vs LSTM vs Bi-LSTM | 45-day lookback sliding window regression on Nifty 50 daily market data. | *Coming Soon* | 🚧 In Progress |
| **04** | **Transformers & Self-Attention** | **NLP & LLM Foundations**<br>Scaled Dot-Product ($Q,K,V$) & Multi-Head | Scratch implementation of Attention mechanics & positional encoding. | *Upcoming* | ⏳ Queued |
| **05** | **RAG Systems & Vector DBs** | **Information Retrieval & GenAI**<br>ChromaDB, Hybrid Retrieval, Reranking | Production document Q&A pipeline with hallucination guardrails. | *Upcoming* | ⏳ Queued |
| **06** | **Autonomous Agentic AI** | **Multi-Agent Systems**<br>LangGraph / CrewAI, Tool Calling | Cyclic state graphs, human-in-the-loop, and external tool execution. | *Upcoming* | ⏳ Queued |

---

## 🏗️ Architecture Standards Followed

Every project in this repository follows strict production and developer documentation standards:
1. **Interactive Execution:** 1-click **Open In Colab** badges with pre-executed inline plots and logs.
2. **Clear Engineering Documentation:**
   * **`README.md`:** Problem statement, architecture diagrams, parameter breakdown tables, verified benchmark metrics, and execution instructions.
3. **Data Integrity:** All stated metrics and benchmarks are 100% verified against executed notebook cells.

---

## 👨‍💻 Connect with Me

* **Author:** Sahil Goury
* **LinkedIn:** [linkedin.com/in/sahil-goury-269107370/](https://www.linkedin.com/in/sahil-goury-269107370/)
* **GitHub:** [@sahilgoury7](https://github.com/sahilgoury7)
