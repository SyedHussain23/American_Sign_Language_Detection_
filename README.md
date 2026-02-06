# 🤟 American Sign Language (ASL) Detection using CNN

This project implements a **beginner-level Convolutional Neural Network (CNN)** to classify **American Sign Language (ASL) hand signs** from images.  
It demonstrates the complete deep learning workflow, including dataset handling, preprocessing, model training, evaluation, and prediction.

---

## 📌 Project Overview

- **Problem Type:** Multi-class image classification  
- **Domain:** Computer Vision  
- **Classes:** 29 (A–Z, SPACE, DELETE, NOTHING)  
- **Model:** Convolutional Neural Network (CNN)  
- **Framework:** TensorFlow / Keras  

---

## 📊 Dataset

- **Source:** ASL Alphabet Dataset (Kaggle)  
- **Total Images:** ~87,000  
- **Classes:** 29 ASL hand sign categories  
- **Image Size:** 64 × 64  
- **Download Method:** `kagglehub`

### Dataset Split
- **Training Set:** ~64%  
- **Validation Set:** ~16%  
- **Test Set:** ~20%  

---

## 🔧 Data Loading & Preprocessing

- Dataset downloaded and extracted using `kagglehub`
- Images loaded using OpenCV (`cv2`)
- All images resized to **64 × 64**
- Labels assigned based on directory structure
- Dataset split into **training, validation, and test sets**

---

## 🧠 Model Architecture

A **Sequential CNN model** was built with the following layers:

- Conv2D (32 filters, ReLU)
- MaxPooling2D
- Conv2D (64 filters, ReLU)
- MaxPooling2D
- Conv2D (128 filters, ReLU)
- MaxPooling2D
- Flatten
- Dense (128 units, ReLU)
- Dropout (0.5)
- Dense (29 units, Softmax)

### Model Configuration
- **Optimizer:** Adam  
- **Loss Function:** Sparse Categorical Crossentropy  
- **Metric:** Accuracy  

---

## 🏋️ Model Training

- **Epochs:** 2  
- **Batch Size:** 32  
- **Validation Data:** Used during training to monitor performance  

This configuration was intentionally kept minimal to demonstrate a **baseline CNN approach**.

---

## 📈 Evaluation Results

- **Test Accuracy:** ~3.45%  
- **Test Loss:** ~3.36  

### Observations
- Accuracy is close to random guessing due to:
  - Large number of classes (29)
  - Very few training epochs
  - No image normalization
  - No data augmentation
  - Simple CNN architecture

This model is intended as a **baseline**, not a high-performance solution.

---

## 🔍 Prediction Demonstration

The trained model was used to predict labels for unseen test images.  
Predictions were visualized by displaying the image along with:

- True label
- Predicted label

---

## 🛠️ Tech Stack

| Tool | Purpose |
|-----|--------|
| Python | Programming |
| TensorFlow / Keras | Deep learning |
| OpenCV | Image processing |
| NumPy | Numerical operations |
| Matplotlib | Visualization |
| Jupyter Notebook | Experimentation |

---

## 🔮 Future Improvements

- Normalize image pixel values
- Train for more epochs
- Apply data augmentation
- Use transfer learning (MobileNet, ResNet, EfficientNet)
- Hyperparameter tuning
- Improve model generalization and accuracy

---

## 👨‍💻 Author

**Syed Hussain Abdul Hakeem**

- LinkedIn: https://www.linkedin.com/in/syed-hussain-abdul-hakeem  
- GitHub: https://github.com/SyedHussain23  

---

## 📄 License

This project is open source and available under the MIT License.

---

## ⭐ Show Your Support

If you found this project useful, consider giving it a ⭐.
