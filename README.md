# Pothole Detection using Hybrid Transformer-CNN Ensemble & Explainable AI

This project focuses on automated road damage assessment using a hybrid deep learning pipeline that combines Vision Transformers (ViT) and Convolutional Neural Networks (CNNs). The system classifies road surface images into **Pothole** and **Normal** categories, addressing the need for reliable infrastructure monitoring while emphasizing transparent and interpretable AI decision-making.

## Project Description

This project presents a robust automated pothole detection system built using a **Hybrid Weighted Ensemble** of modern deep learning architectures. By integrating the global contextual reasoning of Transformers with the local texture sensitivity of CNNs, the system achieves strong generalization across challenging road conditions. To promote trust in automated maintenance workflows, the project incorporates a comprehensive **Explainable AI (XAI)** framework (Grad-CAM, SHAP, and LIME), enabling clear visualization of the features influencing each prediction.

## Key Steps

### 1) Data Loading and Exploration

- Load the pothole detection dataset from Kaggle.
- Verify image quality and class balance (Pothole vs. Normal).
- Visualize representative samples to understand variations in lighting, shadows, wet surfaces, and road markings.
- Prepare stratified splits to support reliable cross-validation.

### 2) Data Preprocessing

- **Albumentations Pipeline**: Apply extensive geometric and photometric augmentations (flips, rotations, brightness/contrast) to simulate real-world road conditions.
- **Normalization**: Use ImageNet mean and standard deviation to ensure compatibility with pretrained backbones.
- **Resizing**: Standardize image resolution (e.g., 224×224 or 384×384) according to architectural requirements.

### 3) Model Architecture

A diverse set of ImageNet-pretrained models is trained to capture complementary visual cues:

- **Vision Transformer (ViT-Base)**: Models long-range dependencies and global road structure.
- **Swin Transformer (Tiny)**: Hierarchical Transformer capturing multi-scale contextual information.
- **EfficientNet-B0**: Lightweight CNN optimized for fine-grained texture analysis.
- **ResNet50**: Deep residual CNN for robust and stable feature extraction.

**Ensemble Strategy:**

- Extract logits or class probabilities from all four models.
- Combine predictions using a **Weighted Soft Voting** scheme.
- Exploit architectural diversity to reduce individual model biases and improve robustness.

### 4) Training Strategy

- **Transfer Learning**: Fine-tune pretrained models to accelerate convergence.
- **Automatic Mixed Precision (AMP)**: Improve training efficiency and reduce memory usage.
- **Learning Rate Scheduling**: Cosine Annealing with warmup to stabilize optimization.
- **Optimizer**: AdamW, well-suited for Transformer-based architectures.
- **Loss Function**: Cross-Entropy Loss with label smoothing to improve generalization.

### 5) Model Evaluation

Models are evaluated using standard and robust metrics:

- **Accuracy**: Overall classification correctness.
- **Precision and Recall**: Important for balancing false alarms and missed potholes.
- **F1-Score**: Balanced performance indicator.
- **Confusion Matrices**: Detailed error analysis.
- **K-Fold Cross-Validation**: Ensures performance stability across data splits.

### 6) Ablation Studies

To analyze the contribution of individual design choices, controlled ablation studies are conducted using **K-fold cross-validation**, modifying one component at a time while keeping the rest of the pipeline unchanged. The ablations examine:

- **Pretraining vs. Training from Scratch** – to assess the impact of ImageNet initialization.
- **Data Augmentation** – to evaluate robustness improvements from augmentation.
- **Backbone Freezing vs. Fine-Tuning** – to study feature adaptability.
- **Input Resolution** – to understand the role of spatial detail.
- **Loss Function Choice** – to measure sensitivity to optimization objectives.

These studies clarify which architectural and training decisions most strongly influence performance and explain the effectiveness of the hybrid Transformer–CNN ensemble.

### 7) Explainable AI (XAI) Framework

Model transparency is ensured through multiple complementary interpretability techniques:

- **Grad-CAM**: Highlights spatial regions influencing CNN predictions.
- **SHAP (SHapley Additive exPlanations)**: Quantifies the contribution of Transformer input regions.
- **LIME (Local Interpretable Model-agnostic Explanations)**: Provides local, perturbation-based explanations independent of model type.

## Results

The system demonstrates strong and consistent pothole detection performance:

**Key Findings:**

- The hybrid ensemble outperforms individual CNN and Transformer models by combining texture-level and contextual cues.
- **Vision Transformers** are particularly effective at distinguishing potholes from visually similar artifacts such as shadows.
- **CNNs** excel at capturing sharp edges and surface irregularities.
- XAI visualizations confirm that predictions are driven by pavement damage characteristics rather than irrelevant background features.

## Dataset

**Source**: [Kaggle – Pothole Detection Dataset](https://www.kaggle.com/datasets/atulyakumar98/pothole-detection-dataset)

**Classes**: Pothole, Normal

**Type**: RGB images of road surfaces

## Dependencies

The project uses the following Python libraries:

```
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

Install dependencies with:

```bash
pip install torch torchvision timm albumentations scikit-learn matplotlib captum shap lime opencv-python
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/sakshammgarg/PotholeDetection.git
cd PotholeDetection
```

2. Download the dataset:
   - Visit the [Kaggle Dataset](https://www.kaggle.com/datasets/atulyakumar98/pothole-detection-dataset).
   - Download and extract the dataset.
   - Organize images as follows:

```
/input
  /pothole-detection-dataset
    /normal
    /potholes
```

## Usage

Run the provided notebook to execute the full training and evaluation pipeline.

The notebook performs the following steps:

1. Initializes the Albumentations preprocessing pipeline.
2. Trains individual models (ViT, Swin, EfficientNet, ResNet) using AMP.
3. Performs ensemble inference.
4. Displays evaluation metrics (Accuracy, F1-score, ROC curves).
5. Generates Grad-CAM, SHAP, and LIME visualizations for selected images.

## Model Selection Guide

**Recommended Choices for Deployment:**

- **Highest Accuracy**: Hybrid Ensemble.
- **Edge or Low-Power Devices**: EfficientNet-B0 or Swin-Tiny.
- **Complex Lighting Conditions**: ViT-Base.
- **Clear Visual Explanations**: ResNet50 with Grad-CAM.

## Advanced Features

Key technical highlights include:

- **Hybrid Deep Learning**: Integration of attention-based and convolutional paradigms.
- **Timm Library Usage**: Access to high-quality pretrained models.
- **Robust Augmentation**: Albumentations for realistic data variation.
- **Mixed Precision Training**: Efficient GPU utilization.
- **Multi-Method XAI**: Cross-verification of explanations.
- **Cosine Warmup Scheduling**: Improved training stability.

## Applications

Potential real-world applications include:

- Autonomous and Assisted Driving Systems
- Smart City Infrastructure Monitoring
- Drone-Based Road Inspection
- Municipal Maintenance Prioritization
- Citizen Reporting and Verification Platforms

## Future Improvements

Planned extensions of this work include:

1. **Object Detection**: Localizing multiple potholes per image.
2. **Severity Estimation**: Predicting pothole depth or damage extent.
3. **Video-Based Inference**: Leveraging temporal consistency from dashcam footage.
4. **Edge Deployment**: Model quantization for mobile and embedded systems.
5. **3D Surface Analysis**: Estimating pothole volume using stereo or depth data.

---

This project demonstrates how hybrid deep learning architectures and explainable AI can be combined to build accurate, robust, and trustworthy systems for automated road infrastructure monitoring.
