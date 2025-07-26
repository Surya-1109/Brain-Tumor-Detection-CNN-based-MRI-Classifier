## 🧠 Brain Tumor Detection Using Deep Learning

This project focuses on the classification of brain MRI images into four categories — **Glioma**, **Meningioma**, **Pituitary Tumor**, and **No Tumor** — using **Convolutional Neural Networks (CNN)** and **Transfer Learning** with **ResNet-18**. The system demonstrates how deep learning can support medical diagnosis by achieving high accuracy in detecting brain tumors from MRI scans.

---

## 📁 Dataset

- **Source**: [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
- **Total Images**: 7,023 labeled MRI scans
- **Categories**: 
  - Glioma Tumor
  - Meningioma Tumor
  - Pituitary Tumor
  - No Tumor
- **Split**:
  - Training Set: 80% (with 20% of that used for validation)
  - Test Set: 20%

---

## 🧪 Methodology

### 🔧 1. Custom CNN Model
- Built from scratch using PyTorch
- 5 convolutional layers with increasing filter sizes: 32 → 256
- ReLU activations, MaxPooling, Batch Normalization
- Fully Connected Layers:
  - FC1: 256×7×7 → 512 + Dropout
  - FC2: 512 → 4 (softmax over classes)
- Optimizer: `Adam` (learning rate = 0.0007)
- Loss Function: `CrossEntropyLoss`
- Early stopping to avoid overfitting

### 🔄 2. Transfer Learning with ResNet-18
- Pretrained on ImageNet
- Modified final fully connected layer for 4-class output
- Two-phase training:
  - **Feature Extraction Phase**: only FC layers trained
  - **Fine-Tuning Phase**: last few layers (layer3, layer4, FC) unfrozen
- Learning rates: 0.001 (initial), 1e-4 (fine-tuning)
- Early stopping applied

---

## 📈 Results

| Model                | Train Accuracy | Validation Accuracy | Test Accuracy |
|---------------------|----------------|---------------------|---------------|
| Custom CNN          | 98.34%         | 97.64%              | 98.17%        |
| ResNet-18 (Initial) | 90.06%         | 91.69%              | 89.32%        |
| ResNet-18 (FT)      | 89.60%         | 91.34%              | 89.09%        |

- The **custom CNN** outperformed both stages of ResNet-18, indicating that task-specific architectures can be more effective in medical imaging than generic pretrained networks.

---

## 📊 Visualizations

| Metric | Custom CNN | ResNet-18 |
|--------|------------|-----------|
| ROC Curve | ✅ | ✅ |
| Confusion Matrix | ✅ | ✅ |
| Loss Curves | ✅ | ✅ |


---

## 🧠 Why This Project Matters

Manual analysis of brain MRI scans is:
- Time-consuming
- Prone to human error
- Requires expert interpretation

This project provides:
- A reproducible and scalable framework for brain tumor classification
- High accuracy and fast inference
- Support for radiologists and clinicians in early tumor detection

---

