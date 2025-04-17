# Insurance Cost Prediction Web App

This project is a **Machine Learning-based web application** built using **Streamlit** to predict medical insurance costs based on user input. It allows users to input personal and medical information and receive an estimated insurance charge instantly.

## Main Features

- Predicts insurance cost based on:
  - Age
  - Gender
  - BMI (Body Mass Index)
  - Number of children
  - Smoking status
  - Region (Northeast, Southeast, Northwest, Southwest)
- Interactive and easy-to-use web interface with Streamlit
- Real-time prediction using a trained regression model


##  Machine Learning Model

- **Algorithm Used:** Linear Regression , OneHotEncoding , LogTransform
- **Target Variable:** `charges`
- **Evaluation Metrics:** MAE, RMSE, R² Score (adjust based on your project)
---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Eng-abdelhamed/insurance-cost-prediction.git
cd insurance-cost-prediction
```
```python
# You have also to run the Streamlit Source Code
streamlit run app.py
```

```bash
insurance-cost-prediction/
│
├── app.py                  # Streamlit app file
├── model.pkl               # Trained ML model
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── insurance.csv           # Dataset (optional)
```
