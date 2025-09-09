# 🩺 Heart Disease Prediction using Machine Learning

This project uses multiple machine learning models to predict the presence of **heart disease** based on patient health attributes. The goal is to provide an interpretable, deployment-ready model that can support early diagnosis and prevention.

---

## 📂 Project Structure

```
HeartDiseasePrediction - ML model/
│
├── models/
│   ├── logistic_regression_model.pkl
│   ├── scaler.pkl
│
├── notebooks/
│   ├── analysis.ipynb
│
├── requirements.txt
├── README.md
```

---

## 📊 Dataset

* Source: **UCI Machine Learning Repository – Heart Disease Dataset**
* Records: \~920 samples
* Features: 14 attributes such as:

  * `age`, `sex`, `cp` (chest pain type), `trestbps` (resting blood pressure),
  * `chol` (cholesterol), `thalach` (max heart rate), `oldpeak` (ST depression),
  * `exang` (exercise-induced angina), and others.
* Target: `0` = No Heart Disease, `1` = Heart Disease

---

## ⚙️ Data Preprocessing

* Encoded categorical variables (e.g., `sex`, `cp`, `fbs`, `restecg`, `exang`, `thal`)
* Scaled numerical features using **StandardScaler**
* Split dataset: **80% training, 20% testing**

---

## 🤖 Models Trained

1. Logistic Regression ✅ (Best Model)
2. Random Forest
3. Support Vector Machine (SVM)
4. XGBoost
5. K-Nearest Neighbors (KNN)
6. Decision Tree
7. Naive Bayes

**Evaluation Metrics**: Accuracy, ROC AUC, Precision, Recall, F1-score

* **Best Accuracy:** Logistic Regression → **76.58%**
* **Best ROC AUC:** Naive Bayes → **0.8465**

---

## 📈 Feature Importance (Logistic Regression)

Top predictive features included:

* **Chest Pain Type (cp)**
* **Maximum Heart Rate (thalach)**
* **Exercise-induced Angina (exang)**
* **ST Depression (oldpeak)**

---

## 🚀 How to Use

### 1. Clone Repository

```bash
git clone https://github.com/YasiruChamithLansakara/HeartDiseasePrediction - ML model.git
cd HeartDiseasePrediction - ML model
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Load Model and Predict

```python
import joblib
import numpy as np

# Load saved model & scaler
model = joblib.load("models/logistic_regression_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Example patient input [age, trestbps, chol, thalach, oldpeak, ... one-hot encoded features ...]
sample = np.array([[55, 130, 250, 150, 1.0, 1, 0, 1, 0, 0, 1, 0, 0]])
sample_scaled = scaler.transform(sample)

# Prediction
print("Prediction:", model.predict(sample_scaled))   # 0 = No disease, 1 = Disease
```

---

## 🔮 Next Steps

* Build a simple **Streamlit web app** for real-time predictions
* Deploy model on **Heroku** or **AWS**
* Test with external medical datasets

---

## 👨‍💻 Author

* **L M Y C Lansakara**
* 📧 [yasiruchamithlansakara@gmail.com](mailto:yasiruchamithlansakara@gmail.com)
* 🌐 \[LinkedIn](https://www.linkedin.com/in/yasiru-chamith-lansakara/)

---
