# 🔬 Breast Cancer Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

An AI-powered Streamlit app that estimates whether a breast tumor is **Benign** or **Malignant** using cytological features from the Breast Cancer Wisconsin dataset.

> **Important (Medical Disclaimer):** This project is for education/demonstration only and is **not medical advice**.
> Model outputs are probabilities and can be wrong. Clinical diagnosis requires qualified professionals and confirmatory tests.

---

## Jump to

- [✨ Features](#-features)
- [🚀 Quick start (PowerShell)](#-quick-start-powershell)
- [🧠 How the prediction works](#-how-the-prediction-works)
- [🧬 Input features](#-input-features)
- [📸 Screenshots](#-screenshots)
- [📂 Project structure](#-project-structure)
- [🧯 Troubleshooting](#-troubleshooting)
- [👨‍💻 Author](#-author)

---

## ✨ Features

- **Interactive dashboard** built with Streamlit.
- **Manual patient input** (numeric fields) and **CSV upload** (batch predictions).
- **Probability output** for both classes + confidence-style visuals.
- **Doctor-style explanation**: translates probabilities into plain-language guidance (rule-based risk tiers + model disagreement note).
- **Model comparison (training tab)**: compares SVM vs Logistic Regression on the Wisconsin diagnostic dataset.

<details>
<summary><b>What makes this README “interactive”?</b></summary>

- Collapsible sections you can open only when you need them
- Copy/paste-ready commands for Windows PowerShell
- Clear “what to do next” paths depending on whether your `.pkl` files exist

</details>

---

## 🚀 Quick start (PowerShell)

### 1) Install dependencies

```powershell
pip install -r requirements.txt
```

### 2) (Optional) Train / regenerate model files

If you don’t have `model.pkl` or `scaler.pkl`, run:

```powershell
python train_model.py
```

### 3) Run the Streamlit app

```powershell
streamlit run app.py
```

Streamlit will typically open the browser automatically. If it doesn’t, look in the terminal output for the local URL (commonly `http://localhost:8501`).

<details>
<summary><b>Tip: Create a virtual environment (recommended)</b></summary>

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

</details>

---

## 🧠 How the prediction works

<details>
<summary><b>Models used</b></summary>

This app includes:

- **SVM with GridSearchCV** (optimized)
- **Logistic Regression** (baseline)

In the prediction UI, both models produce probabilities, and the app also shows a **combined malignant probability** (simple average) for the doctor-style explanation.

</details>

<details>
<summary><b>Doctor-style explanation</b></summary>

The explanation text is **rule-based** (not a medical guideline): it takes the probability + threshold and returns:

- a short **Summary**
- a longer **Details** paragraph
- and **Recommended next steps**

If two models disagree strongly, it will add an extra note encouraging clinical follow-up.

</details>

---

## 🧬 Input features

The models use 10 mean features:

- Radius, Texture, Perimeter, Area, Smoothness
- Compactness, Concavity, Concave Points, Symmetry, Fractal Dimension

<details>
<summary><b>CSV upload format</b></summary>

Your uploaded CSV should include columns matching the training feature names.

If your CSV is missing columns, you’ll see an error during prediction.

</details>

---

## 📸 Screenshots

<details>
<summary><b>Add your screenshots here</b></summary>

- Put images in a folder like `assets/` and link them here.

Example:

```text
assets/
  training_tab.png
  prediction_tab.png
```

</details>

---

## 📂 Project structure

```text
Bio2_Mini_Project/
├── app.py                       # 🖥️ Main Streamlit application
├── model_utils.py               # 🧰 Shared utilities (incl. explanations)
├── train_model.py               # ⚙️ Script to train and save the ML model
├── model.pkl                    # 🧠 Saved model artifact
├── scaler.pkl                   # ⚖️ Saved scaler artifact
├── pages/
│   ├── 1_⚙️_Model_Configuration.py
│   └── 2_🧬_Patient_Input.py
├── requirements.txt             # 📦 Dependencies
└── README.md                    # 📄 You are here
```
</details>

---
