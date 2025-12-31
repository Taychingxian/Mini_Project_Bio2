# 🔬 Breast Cancer Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

An AI-powered web application designed to predict the likelihood of a breast tumor being **Benign** or **Malignant** based on cytological features. This tool utilizes a **Random Forest Classifier** trained on the Breast Cancer Wisconsin dataset.

---

## ✨ Features

*   **Interactive Dashboard**: User-friendly interface built with Streamlit.
*   **Real-time Prediction**: Instant classification results as you adjust feature sliders.
*   **Visual Insights**: Displays prediction probabilities and key feature values.
*   **Machine Learning**: Powered by a robust Random Forest model.
*   **Data Preprocessing**: Includes automated feature scaling for accurate predictions.

---

## 🧠 Model Information

The system uses a **Random Forest Classifier** trained on the **Breast Cancer Wisconsin (Diagnostic) Data Set**.

*   **Algorithm**: Random Forest (Ensemble Learning)
*   **Input Features**: 10 key cytological attributes (Mean values):
    *   Radius, Texture, Perimeter, Area, Smoothness
    *   Compactness, Concavity, Concave Points, Symmetry, Fractal Dimension
*   **Target Classes**:
    *   🟢 **Benign** (Non-cancerous)
    *   🔴 **Malignant** (Cancerous)

---

## 🚀 Quick Start

### Prerequisites

Ensure you have Python installed.

### Installation

1.  **Clone the repository** (or download the files):
    ```bash
    cd Bio2_Mini_Project
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  **Train the Model** (Optional, if `.pkl` files are missing):
    ```bash
    python train_model.py
    ```
    *This will generate `model.pkl` and `scaler.pkl`.*

2.  **Run the Application**:
    ```bash
    streamlit run app.py
    ```

3.  **Access the App**:
    The application will open automatically in your browser at `http://localhost:8501`.

---

## 📂 Project Structure

```text
Bio2_Mini_Project/
├── app.py              # 🖥️ Main Streamlit application
├── train_model.py      # ⚙️ Script to train and save the ML model
├── model.pkl           # 🧠 Trained Random Forest model (generated)
├── scaler.pkl          # ⚖️ Feature scaler (generated)
├── requirements.txt    # 📦 List of dependencies
└── README.md           # 📄 Project documentation
```

---

## 📊 Performance

The model is evaluated using standard metrics:
*   **Accuracy**
*   **Precision & Recall**
*   **Confusion Matrix**

*(Run `train_model.py` to see the latest performance metrics in the terminal.)*

---
