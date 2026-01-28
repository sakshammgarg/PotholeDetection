# Pothole Detection using Hybrid Transformer-CNN Ensemble & Explainable AI

This project focuses on automated road damage assessment using a hybrid deep learning pipeline that combines Vision Transformers (ViT) and Convolutional Neural Networks (CNNs). The system accurately classifies road surface images into **Potholes** and **Normal** categories, addressing the need for robust infrastructure monitoring and transparent AI decision-making.

## Project Description

This project aims to build a resilient automated pothole detection system using a **Weighted Ensemble** of state-of-the-art architectures. By integrating the global context capabilities of Transformers with the local texture extraction of CNNs, the model achieves superior generalization. To ensure trust in automated maintenance systems, the project implements a comprehensive **Explainable AI (XAI)** suite (Grad-CAM, SHAP, LIME) to visualize exactly where the model detects damage.

## Key Steps

### 1) Data Loading and Exploration

* Load the Pothole Detection dataset from Kaggle
* Inspect image integrity and label distribution (Pothole vs. Normal)
* Visualize random samples to understand surface variations (lighting, wet roads, shadows)
* Split data into stratified folds for robust Cross-Validation

### 2) Data Preprocessing

* **Albumentations Pipeline**: Heavy augmentation (flips, rotations, brightness contrast) to simulate real-world road conditions.
* **Normalization**: Apply ImageNet mean/std normalization for transfer learning compatibility.
* **Resizing**: Standardize inputs (e.g., 224x224 or 384x384) specific to each model architecture.

### 3) Model Architecture

Train a diverse set of ImageNet-pretrained models to capture different image features:

* **Vision Transformer (ViT-Base)**: Captures long-range dependencies and global road structure.
* **Swin Transformer (Tiny)**: Hierarchical transformer for multi-scale feature extraction.
* **EfficientNet-B0**: Optimized CNN for identifying fine-grained textures efficiently.
* **ResNet50**: Deep residual network for robust feature extraction.

**Ensemble Strategy:**

* Extract logits/probabilities from all four fine-tuned models.
* Apply a **Weighted Soft Voting** mechanism to combine predictions.
* Leverage the diversity of Transformers and CNNs to correct individual model errors.

### 4) Training Strategy

* **Transfer Learning**: Fine-tuning pre-trained weights to converge faster.
* **Mixed-Precision Training (AMP)**: Reduced memory usage and faster training via `torch.cuda.amp`.
* **Scheduler**: Cosine Annealing with Warmup to prevent local minima stagnation.
* **Optimizer**: AdamW (optimized for Transformers).
* **Loss Function**: Cross-Entropy Loss with Label Smoothing.

### 5) Model Evaluation

Evaluate models using standard computer vision metrics:

* **Accuracy**: Overall correctness of road assessment.
* **Precision/Recall**: Critical for minimizing false alarms (maintenance cost) and missed potholes (safety risk).
* **F1-Score**: Harmonic mean for balanced performance.
* **Confusion Matrices**: Detailed breakdown of misclassifications.
* **K-Fold Cross-Validation**: ensuring the model generalizes across different data splits.

### 6) Ablation Studies

To better understand the contribution of individual components, we conduct controlled ablation studies using **5-fold cross-validation**, where only one factor is modified at a time while keeping the remaining training protocol unchanged. The ablations analyze:

- **Pretraining vs. Training from Scratch** – to quantify the benefit of ImageNet initialization.
- **Data Augmentation** – to assess robustness gains from geometric and photometric transformations.
- **Backbone Fine-Tuning vs. Freezing** – to evaluate the necessity of deep feature adaptation.
- **Input Resolution** – to study the effect of spatial detail on pothole recognition.
- **Loss Function Choice** – to examine performance sensitivity to different optimization objectives.

These studies isolate which architectural and training decisions most strongly influence performance and provide deeper insight into why the hybrid Transformer–CNN ensemble achieves superior results.

### 7) Explainable AI (XAI) Framework

Ensure transparency through a multi-method interpretability suite:

* **Grad-CAM**: Generates heatmaps for CNNs to show "where" the model is looking.
* **SHAP (SHapley Additive exPlanations)**: Game-theoretic approach to explain Transformer token importance.
* **LIME (Local Interpretable Model-agnostic Explanations)**: Perturbation-based explanation for model-agnostic verification.

## Results

The project demonstrates highly effective road damage detection with rigorous validation:

**Key Findings:**

* Hybrid Ensemble outperforms individual CNN or ViT models by capturing both texture and structure.
* **ViT** excels at distinguishing complex road scenes (e.g., shadows vs. holes).
* **CNNs** excel at detecting sharp edges and crack textures.
* XAI visualizations confirm the model focuses on the actual pavement defects rather than background noise (e.g., grass, cars).

## Dataset

**Source**: [Kaggle - Pothole Detection Dataset](https://www.kaggle.com/datasets/atulyakumar98/pothole-detection-dataset)

**Classes**: Pothole, Normal

**Type**: RGB Images of road surfaces

## Dependencies

The project requires the following Python libraries:

```bash
numpy
pandas
torch
torchvision
timm
albumentations
scikit-learn
matplotlib
captum
shap
lime
opencv-python

```

Install the dependencies using:

```bash
pip install torch torchvision timm albumentations scikit-learn matplotlib captum shap lime opencv-python

```

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/sakshammgarg/PotholeDetection.git
cd PotholeDetection
```

2. **Download the dataset**
* Visit [Kaggle Dataset](https://www.kaggle.com/datasets/atulyakumar98/pothole-detection-dataset)
* Download and extract the dataset
* Organize images into the standard directory structure:

```
/input
  /pothole-detection-dataset
    /normal
    /potholes
```

## Usage

Run the notebook to explore the complete analysis and training pipeline.

The notebook will:

1. Initialize the Albumentations preprocessing pipeline.
2. Train individual models (ViT, Swin, EfficientNet, ResNet) with AMP.
3. Perform Ensemble inference on the test set.
4. Display evaluation metrics (Accuracy, F1, ROC Curves).
5. Generate **Grad-CAM**, **SHAP**, and **LIME** visualizations for specific test images.

## Model Selection Guide

**For Deployment Scenarios:**

* **Best Accuracy**: Hybrid Ensemble (ViT + CNN combined).
* **Edge Devices (Drones/Dashcams)**: EfficientNet-B0 or Swin-Tiny (Low latency).
* **Complex Environments**: ViT-Base (Better handling of shadows/lighting).
* **Interpretability**: ResNet50 with Grad-CAM (Cleanest heatmaps).

## Advanced Features

This comprehensive implementation includes:

* **Hybrid Deep Learning**: Merging two distinct paradigms (Attention mechanisms & Convolutions).
* **Timm Library Integration**: Access to state-of-the-art pre-trained weights.
* **Robust Preprocessing**: Albumentations for geometric and color-space augmentation.
* **Automatic Mixed Precision**: Optimizing training speed on GPUs.
* **Triangulated XAI**: Using three different explanation methods to verify model trust.
* **Cosine Warmup**: Advanced learning rate scheduling for training stability.

## Applications

Practical use cases for this pothole detection system:

* **Autonomous Vehicles**: Real-time road scanning for navigation adjustment.
* **Smart Cities**: Automated surveying using municipal garbage trucks or buses.
* **Drone Surveillance**: Aerial inspection of highway infrastructure.
* **Maintenance Prioritization**: Categorizing road damage severity for repair scheduling.
* **Citizen Reporting Apps**: Validating user-uploaded photos of road damage.

## Future Improvements

Potential enhancements for even better results:

1. **Object Detection**: Switching to YOLO/Faster-RCNN to localize multiple potholes per image.
2. **Severity Estimation**: Regression model to predict the depth/volume of the pothole.
3. **Video Inference**: Temporal consistency checks for dashcam video feeds.
4. **Edge Optimization**: Quantization (INT8) for deployment on mobile phones.
5. **3D Reconstruction**: Using stereo vision to estimate pothole volume for filler material calculation.

---

This project demonstrates how **Hybrid Architectures** and **Explainable AI** can be combined to build accurate, robust, and transparent infrastructure monitoring systems.
