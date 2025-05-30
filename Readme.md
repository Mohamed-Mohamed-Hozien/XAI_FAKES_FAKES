# Explainable AI for Chest X-ray Disease Classification

This project explores the application of Explainable Artificial Intelligence (XAI) techniques to Machine Learning (ML) and Deep Learning (DL) models for multi-label classification of thoracic diseases using chest X-ray (CXR) images. The goal is to improve diagnostic trustworthiness by making AI predictions interpretable and clinically transparent.

## 📁 Project Structure

- `notebook.ipynb`: Main notebook with end-to-end data preprocessing, model training, evaluation, and interpretability visualizations.
- `Fakes Fakes Research Paper v1.pdf`: Research documentation outlining methodology, model comparisons, results, and insights.

## 📊 Datasets

We used two publicly available datasets on Kaggle derived from the **NIH ChestX-ray14** dataset:

- **Original Dataset**: [NIH Chest X-rays (Full)](https://www.kaggle.com/datasets/nih-chest-xrays/data)
- **Balanced Dataset**: [Balanced & Resized NIH X-rays](https://www.kaggle.com/datasets/rahulogoel/nih-balanced-and-resized-chest-x-rays)

> ✅ **Tip:** You can open and run the notebook directly in [Kaggle Notebooks](https://www.kaggle.com/code) for a GPU-enabled environment.

## 🧪 Methodology

### 🔧 Preprocessing

- Resized all images to 224×224 or 256×256 pixels depending on model requirements.
- Excluded outliers (e.g. patient age > 120).
- Converted multi-label disease annotations using `MultiLabelBinarizer`.
- Split the data into training, validation, and testing sets using patient-wise stratification.

### 🧠 Models Used

The following models were implemented and compared:

- **Machine Learning**: Kernel SVM (Linear, Polynomial, RBF, Sigmoid)
- **CNNs**: DenseNet121, MobileNetV2, CustomCNN
- **Transformers**: SwinCheX (Swin Transformer backbone)
- **Autoencoder-Based**: AE-CNN, MLRFNet
- **Attention-Based CNNs**: CRAL with ResNet/DenseNet
- **Interpretability-Focused**: MAMMI with MedCLIP

### 🏗️ Interpretability Techniques

We employed various XAI tools to visualize and understand predictions:

- **Grad-CAM** – Highlighted regions of model focus
- **LIME** – Local image perturbation analysis
- **SHAP** – Game-theory-based global explanations
- **Eigen-CAM** – Gradient-free heatmaps
- **Saliency Maps** – Pixel-level sensitivity analysis
- **DeepDream** – Neuron activation pattern enhancement

## 📈 Evaluation Metrics

- **AUC (Area Under ROC Curve)** for all 14 diseases
- Accuracy, Precision, Recall, F1-Score for multi-label performance
- Visualization-based model trust assessment

## 🧪 Results

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
| CRAL                   | 0.812        | 0.607     |
| MAMMI (XAI-first)      | 0.881        | 0.816     |

## 🧩 Challenges

- **Label imbalance**: Rare diseases like Hernia were underrepresented.
- **Co-occurring conditions**: Complex label overlaps made classification difficult.
- **Image resolution**: Trade-offs between quality and computation required careful handling.
- **Hardware limits**: Some models could not be trained for the full epoch range.

## 🏁 Conclusion

The integration of XAI tools significantly improved the interpretability of thoracic disease classification models. While models like AE-CNN and DenseNet121 delivered high AUC scores, methods like LIME and Grad-CAM helped verify clinical relevance. Future improvements could include training on larger balanced datasets and incorporating hybrid CNN-Transformer architectures.

## 📚 References

- [NIH ChestX-ray14 Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data)
- [Balanced NIH X-ray Dataset](https://www.kaggle.com/datasets/rahulogoel/nih-balanced-and-resized-chest-x-rays)
- Papers on Grad-CAM, LIME, SHAP, MAMMI, CheXNet, and Swin Transformers as cited in the full PDF.

## ✍️ Authors

- **Yusuf Tamer** – [Email](mailto:s-yusuf.sahab@zewailcity.edu.eg)  
- **Mohamed Mohamed** – [Email](mailto:s-mohamed.hozien@zewailcity.edu.eg)  
- **Ziad Shaaban** – [Email](mailto:s-ziad.amer@zewailcity.edu.eg)  
- **Mohammed Taha** – [Email](mailto:s-mohammed.taha@zewailcity.edu.eg)

Course: **Explainability and Interpretability in AI**, Zewail City of Science and Technology
