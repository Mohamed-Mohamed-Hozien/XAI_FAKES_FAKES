# Explainable AI for Chest X-ray Disease Classification

This project explores the application of **Explainable Artificial Intelligence (XAI)** techniques to Machine Learning (ML) and Deep Learning (DL) models for multi-label classification of thoracic diseases using chest X-ray (CXR) images. The goal is to improve diagnostic trustworthiness by making AI predictions interpretable and clinically transparent.

## 📁 Project Structure

This repository includes multiple Jupyter notebooks, each covering one of the 12 implemented models or a specific experiment. There is no single “main” notebook; please refer to the notebook titles to locate each model or technique.

| Notebook Filename                            | Model or Purpose                      |
|----------------------------------------------|----------------------------------------|
| `SVM-V1.ipynb`, `SVM-V2.ipynb`               | Kernel SVM (Linear, RBF, Polynomial)   |
| `multi-label-classification.ipynb`           | Multi-label pipeline setup             |
| `XAI Techniques.ipynb`                       | General XAI (Grad-CAM, LIME, SHAP)     |
| `ae-cnn-taha.ipynb`                          | AE-CNN (Autoencoder + Classifier)      |
| `customcnn_Taha.ipynb`                       | Custom CNN                             |
| `residual_Taha.ipynb`                        | Residual Attention CNN                 |
| `mammi-yusuf-sahab.ipynb`                    | MAMMI (MedCLIP + Neuron Interpretation)|
| `swinchex-fakes.ipynb`                       | SwinCheX (Swin Transformer backbone)   |
| `densenet121_finetune.ipynb`                 | Fine-tuned DenseNet121                 |
| `mobilenetv2_finetune.ipynb`                 | Fine-tuned MobileNetV2                 |
| `mobilelungnetv2.ipynb`                      | MobileLungNetV2 (Custom MobileNet head)|
| `mlrfnet.ipynb`                              | MLRFNet (Multi-Level Residual Net)     |
| `fakes-EDA-Data-Exploration.ipynb`           | Data exploration and preprocessing     |

> ✅ **Tip:** You can open and run these notebooks directly in [Kaggle Notebooks](https://www.kaggle.com/code) with GPU support.

## 📊 Datasets

We used two publicly available Kaggle datasets derived from the NIH ChestX-ray14 dataset:

- **Original Dataset**: [NIH Chest X-rays](https://www.kaggle.com/datasets/nih-chest-xrays/data)
- **Balanced & Resized**: [Rahul Goel’s Preprocessed NIH Set](https://www.kaggle.com/datasets/rahulogoel/nih-balanced-and-resized-chest-x-rays)

## 🧪 Methodology

### 🔧 Preprocessing

- Resized images (224×224 or 256×256)
- Removed outliers (e.g. patient age > 120)
- Multi-label binarization using `MultiLabelBinarizer`
- Patient-wise train-validation-test split to prevent leakage

### 🧠 Models Implemented

We implemented and evaluated the following 12 models:

1. **Kernel SVM** (Linear, RBF, Polynomial, Sigmoid)
2. **Fine-tuned DenseNet121**
3. **Fine-tuned MobileNetV2**
4. **MobileLungNetV2** (custom MobileNet head)
5. **Custom CNN** (built from scratch)
6. **AE-CNN** (Autoencoder + classifier)
7. **Residual CNN** (CRAL variant)
8. **MLRFNet** (Multi-Level Residual Fusion Network)
9. **SwinCheX** (Swin Transformer-based)
10. **MAMMI** (MedCLIP + neuron interpretation)
11. **CRAL Attention Network** (ResNet/DenseNet with attention)
12. **Explainability-First DenseNet** (with Grad-CAM, Saliency, LIME)

### 🏗️ XAI Techniques

- **Grad-CAM**
- **LIME**
- **SHAP**
- **Eigen-CAM**
- **Saliency Maps**
- **DeepDream**
- **Neuron Concept Attribution (MAMMI)**

## 📈 Evaluation Metrics

- **AUC** (Area Under ROC Curve) for each label
- Accuracy, Precision, Recall, and F1-Score
- Visual and clinical validation of interpretability

## 🧪 Results Summary

| Model                  | Original AUC | Our AUC   |
|------------------------|--------------|-----------|
| Kernel SVM             | 0.92 (acc.)  | 0.40      |
| DenseNet121            | 0.877        | 0.75      |
| MobileLungNetV2        | 0.923        | 0.71      |
| MobileNetV2            | 0.810        | 0.72      |
| SwinCheX               | 0.810        | 0.795     |
| AE-CNN                 | 0.843        | 0.9987    |
| MLRFNet                | 0.853        | 0.519     |
| Custom CNN             | 0.95 (acc.)  | 0.42      |
| CRAL Network           | 0.812        | 0.607     |
| Residual CNN           | 0.8136       | 0.6179    |
| MAMMI                  | 0.881        | 0.816     |
| Explainability-First   | —            | Varied by model |

## 🧩 Challenges

- Severe class imbalance for rare diseases like Hernia
- Multi-label dependency and overlap
- Limited GPU time in Kaggle/Colab
- Some papers/models implemented in PyTorch required re-implementation in TensorFlow

## 🏁 Conclusion

This project demonstrates that integrating XAI techniques (e.g., Grad-CAM, SHAP, MAMMI) into thoracic disease classification workflows improves transparency and supports clinical trust. While models like AE-CNN and DenseNet121 performed well, their predictions became meaningfully interpretable through layered explainability tools.

## 📚 References

- [NIH ChestX-ray14 Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data)
- [Balanced NIH Dataset by Rahul Goel](https://www.kaggle.com/datasets/rahulogoel/nih-balanced-and-resized-chest-x-rays)
- Referenced literature includes CheXNet, Swin Transformers, MedCLIP, and explainability surveys.

## ✍️ Authors

- **Yusuf Tamer** – [Email](mailto:s-yusuf.sahab@zewailcity.edu.eg)  
- **Mohamed Mohamed** – [Email](mailto:s-mohamed.hozien@zewailcity.edu.eg)  
- **Ziad Shaaban** – [Email](mailto:s-ziad.amer@zewailcity.edu.eg)  
- **Mohammed Taha** – [Email](mailto:s-mohammed.taha@zewailcity.edu.eg)

Course: **Explainability and Interpretability in AI**, Zewail City of Science and Technology
