# 🧠 Convolutional Neural Networks from Scratch — PyTorch

A complete hands-on implementation of CNNs covering spatial geometry, robustness analysis, feature visualization, and advanced optimization techniques. Built using PyTorch on MNIST and CIFAR-10 datasets.

---

## 📌 What's Inside

| Notebook | Topic |
|----------|-------|
| `Q1_Geometry_of_Convolutions.ipynb` | Spatial dimension math + parameter explosion analysis |
| `Q2_CNN_vs_FCNN_Experiment.ipynb` | CNN vs FCNN robustness on translated images |
| `Q3_Feature_Extraction_and_Visualization.ipynb` | Filter gallery + activation maps |
| `Q4_Optimization_and_Robustness.ipynb` | BatchNorm, Internal Covariate Shift, Data Augmentation |

---

## 🔍 Part 1 — Geometry of Convolutions

Manually calculated output shapes at every layer before writing a single line of code.

- Input: `64 x 64 x 3`
- Architecture: 3 Conv (3×3) + 2 MaxPool (2×2, stride 2) + 1 FC
- Verified using `print(x.shape)` at every step
- Total parameters: **115,754**

**Key finding:** Removing pooling layers causes a **23x parameter explosion** in the FC layer (92K → 2.15M), leading to overfitting and GPU memory exhaustion.

---

## 🥊 Part 2 — CNN vs FCNN Robustness Duel

Trained both models on MNIST, then tested on **Shifted MNIST** (every image shifted 4px right using `torch.roll`).

| Model | Original Accuracy | Shifted Accuracy | Drop |
|-------|:-----------------:|:----------------:|:----:|
| FCNN  | 98.07% | 35.74% | **62.33%** |
| CNN   | 98.73% | 61.63% | **37.10%** |

**Why CNN wins:** Weight sharing gives CNNs translation equivariance — the same filter detects edges regardless of where they appear. FCNNs have position-bound weights so a shifted digit looks completely new.

---

## 🎨 Part 3 — Feature Extraction and Visualization

### Filter Gallery
Visualized all first-layer kernels for both datasets:
- **MNIST filters** — simple edge detectors (horizontal, vertical, diagonal gradients)
- **CIFAR-10 filters** — rich Gabor-like patterns with colour opponency (teal-orange, green-purple)

### Activation Maps
Compared activations after conv1 (early) vs conv3 (deep):

| Layer | What it sees |
|-------|-------------|
| conv1 (early) | Every edge and texture across the whole image |
| conv3 (deep) | Sparse, semantic — object region only |

This confirms the CNN hierarchy: **edges → textures → parts → objects**

---

## ⚙️ Part 4 — Advanced Optimization

### BatchNorm vs No BatchNorm (8-layer deep CNN)
Tracked mean and variance of the 5th layer activations across 500 batches:

- **Without BN:** activation mean drifts, variance fluctuates — clear Internal Covariate Shift
- **With BN:** mean stays near 0, variance stays constant — ICS eliminated

### Data Augmentation (`RandomRotation(30)` + `ColorJitter`)
- Training accuracy slightly lower with augmentation (harder examples)
- Test accuracy higher (better generalization, less overfitting)
- Train-test gap is smaller → augmentation is effective regularization

---

## 📊 Results Summary

| Part | Key Result |
|------|-----------|
| Geometry | 23x parameter explosion without pooling |
| Robustness | CNN 1.68x more robust than FCNN on shifted images |
| Filters | RGB filters show Gabor + colour opponency; grayscale filters show only edges |
| BatchNorm | Eliminates activation drift across 500 training batches |
| Augmentation | Reduces overfitting, improves test accuracy |

---

## 🚀 How to Run

### Option 1 — Google Colab (recommended)
Open any notebook directly:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/cnn-from-scratch-pytorch/blob/main/Q1_Geometry_of_Convolutions.ipynb)

> Replace `YOUR_USERNAME` with your GitHub username

### Option 2 — Local
```bash
git clone https://github.com/YOUR_USERNAME/cnn-from-scratch-pytorch.git
cd cnn-from-scratch-pytorch
pip install torch torchvision matplotlib numpy
jupyter notebook
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-red?logo=pytorch)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)

- **Framework:** PyTorch 2.10
- **Datasets:** MNIST, CIFAR-10 (resized to 64×64 as Tiny-ImageNet-10 substitute)
- **Environment:** Google Colab (T4 GPU)

---

## 📁 Folder Structure

```
cnn-from-scratch-pytorch/
│
├── Q1_Geometry_of_Convolutions.ipynb
├── Q2_CNN_vs_FCNN_Experiment.ipynb
├── Q3_Feature_Extraction_and_Visualization.ipynb
├── Q4_Optimization_and_Robustness.ipynb
│
├── results/
│   ├── q1_shapes_output.png
│   ├── q2_shifted_mnist.png
│   ├── q2_robustness_duel.png
│   ├── q3_mnist_filters.png
│   ├── q3_tin_filters.png
│   ├── q3_activation_maps_mnist.png
│   ├── q3_activation_maps_tin.png
│   └── q4_batchnorm_experiment.png
│
└── README.md
```

---

## 📸 Sample Results

### Shifted MNIST — Original vs Shifted 4px Right
> *(add your shifted_mnist.png here)*

### Filter Gallery — MNIST vs CIFAR-10
> *(add your filter images here)*

### Activation Maps — Early vs Deep Layer
> *(add your activation map images here)*

---

## 📝 License

This project is open source under the [MIT License](LICENSE).

---

⭐ If you found this useful, give it a star!
