# ECG

# ❤️ Diagnosis of ECG with a Low-Complexity Deep Learning Model

**Author**: [Soroush Soltanizadeh](https://www.linkedin.com/in/soroush-soltanizadeh-1136892b6/)
**Google Scholar**: [Profile](https://scholar.google.com/citations?user=ARKNJYwAAAAJ&hl=en)

---

## 📌 Overview

Accurate and early detection of cardiac arrhythmias using **Electrocardiogram (ECG)** signals is vital in modern healthcare. This project introduces a **low-complexity 1D Convolutional Neural Network (CNN)** for ECG signal classification using the **MIT-BIH dataset**. To enhance feature representation, **Discrete Wavelet Transform (DWT)** is applied to the raw ECG signals prior to training. The proposed model is lightweight and suitable for **edge devices** and **real-time wearable applications**.

---

## 🧪 Research Objective

The key objective is to design a **computationally efficient CNN** that maintains **high classification accuracy** for ECG-based heart disease diagnosis, especially under constraints relevant to **embedded medical systems**.

---

## 📁 Dataset

* **Name**: MIT-BIH Arrhythmia Dataset
* **Format**: CSV (`MIT_BIH dataset.csv`)
* **Target Column**: `target` (0 = Normal, 1 = Abnormal)
* **Input Features**: Extracted ECG signal features

---

## 🌀 Signal Preprocessing

To extract meaningful patterns from ECG signals:

* **Wavelet Transform** is applied using `pywt.dwt` with `'db2'` wavelet
* The **approximation** and **detail** coefficients are concatenated to form the final feature vector
* Features are standardized using `StandardScaler`

```python
def apply_dwt(data, wavelet='db2'):
    coeffs = pywt.dwt(data, wavelet)
    return np.hstack(coeffs)
```

---

## 🧠 Model Architecture

### 1D CNN Summary:

* **Conv1D**: 8 filters, kernel size = 2, padding = "same", activation = ReLU
* **MaxPooling1D**: pool size = 2
* **Flatten**
* **Dropout**: 50%
* **Dense**: 1 neuron, sigmoid activation

```python
Input Shape   → (n_features, 1)
Conv1D        → filters=8, kernel=2, activation='relu'
MaxPooling1D  → pool_size=2
Dropout       → rate=0.5
Dense         → units=1, activation='sigmoid'
```

---

## ⚙️ Training Details

* **Optimizer**: Adam (learning rate = 0.01)
* **Loss Function**: Binary Crossentropy
* **Evaluation Method**: 5-Fold Cross-Validation
* **Batch Size**: 64
* **Epochs**: 20

---

## 📊 Results

| Metric                      | Value            |
| --------------------------- | ---------------- |
| **Mean Accuracy**           | **≈{:.2f}%**     |
| **Standard Deviation**      | **≈{:.2f}%**     |
| **Model Complexity (CCNN)** | ≈ *(calculated)* |

> 💡 The model demonstrates competitive performance while maintaining **low computational complexity**, making it ideal for **real-time monitoring systems**.

---

## 📈 Model Complexity

The complexity is estimated using the formula:

```python
CCNN = ni * nf * nk * (ns + 2*npad - dilation*(nk-1) - 1/stride + 1)
```

Where:

* `ni` = number of input features after DWT
* `nf` = number of filters
* `nk` = kernel size
* `ns` = number of time steps
* `npad` = padding size

---

## 🩺 Applications

* **Wearable ECG Monitors**
* **Remote Patient Monitoring**
* **Smart Clinics**
* **IoT-Based Heart Health Systems**

---

## 🚀 Getting Started

### ✅ Requirements

```bash
pip install tensorflow numpy pandas pywt scikit-learn
```

### ▶️ Run the Model

```bash
python ecg_diagnosis_cnn.py
```

---

## 📚 Future Work

* Explore multi-class ECG classification (e.g., AFib, PVC)
* Integrate explainability methods (e.g., Grad-CAM, LIME)
* Real-time inference on microcontrollers (e.g., Arduino, Raspberry Pi)
* Enhance feature engineering with advanced signal processing techniques

---

## 📜 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute.
