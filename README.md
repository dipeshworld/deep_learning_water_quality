# 💧 Water Quality Prediction using Deep Learning

## 📌 Overview

This project leverages **Deep Learning (Neural Networks)** to analyze water quality data from the **Central Pollution Control Board (CPCB), India** and predict:

* 📈 **Water Quality Index (WQI)** → Regression problem
* 🏷️ **Water Quality Classification** (Good / Poor / Unsuitable) → Classification problem

Access to clean water is critical, and this project aims to assist in **data-driven decision-making for water quality monitoring**.

---

## 📂 Dataset Description

The dataset contains water quality measurements collected across India for multiple years (2019–2022).

### 🔑 Features

**Geographical Information**

* Well_ID
* State, District, Block, Village
* Latitude, Longitude
* Year

**Water Quality Parameters**

* pH
* Electrical Conductivity (EC)
* Carbonates (CO3)
* Bicarbonates (HCO3)
* Chlorides (Cl)
* Sulfates (SO4)
* Nitrates (NO3)
* Total Hardness (TH)
* Calcium (Ca)
* Magnesium (Mg)
* Sodium (Na)
* Potassium (K)
* Fluoride (F)
* Total Dissolved Solids (TDS)

---

## 🎯 Target Variables

* **Water Quality Index (WQI)** → Continuous value
* **Water Quality Classification** → Categorical label

---

## 🧠 Model Architecture

Two separate Deep Learning models were built:

### 🔹 Regression Model (WQI)

* Dense Neural Network
* ReLU activations
* Dropout for regularization

### 🔹 Classification Model

* Dense Neural Network
* Softmax output layer

---

## ⚙️ Tech Stack

* Python 🐍
* TensorFlow / Keras
* Scikit-learn
* Pandas & NumPy

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/water-quality-dl.git
cd water-quality-dl
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the training script:

```bash
python main.py
```

---

## 📊 Evaluation Metrics

### Regression

* **R² Score**

### Classification

* **Accuracy**
* **F1 Score**

---

## 📈 Sample Output

```
===== REGRESSION RESULTS =====
R² Score: 0.91

===== CLASSIFICATION RESULTS =====
Accuracy: 0.88
F1 Score: 0.87
```

---

## 📁 Project Structure

```
water-quality-dl/
│
├── data/
│   └── water_quality.csv
│
├── models/
│   ├── wqi_regression_model.h5
│   └── water_quality_classifier.h5
│
├── main.py
├── requirements.txt
└── README.md
```

---

## 🔍 Key Features

* Handles missing values
* Feature scaling using StandardScaler
* Label encoding for classification
* Early stopping to prevent overfitting
* Separate optimized models for regression & classification

---

## 💡 Future Improvements

* 🌍 Geo-spatial visualization (map-based insights)
* ⚡ XGBoost / Ensemble models for better tabular performance
* 🔍 SHAP explainability
* 🌐 Deploy using Streamlit or FastAPI
* 🧠 Hyperparameter tuning (Optuna)

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork this repo and submit a PR.

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

* Central Pollution Control Board (CPCB), India
* Open-source ML community

---

## 📬 Contact

If you have any questions or suggestions, feel free to reach out!

---
