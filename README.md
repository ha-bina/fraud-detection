# Fraud Detection Project

This repository provides a complete pipeline for fraud detection, including data preprocessing, feature engineering, and model training/evaluation. The project is modular and organized for clarity and reproducibility.

---

## Project Structure

```
fraud-detection/
├── data/
│   ├── raw/
│   │   ├── Fraud_Data.csv
│   │   └── IpAddress_to_Country.csv
│   └── processed/
│       ├── creditcard_processed.csv
│       └── fraud_data_processed.csv
├── notebooks/
│   ├── fraud_notebook.ipynb         # Data cleaning & feature engineering
│   └── Model training.ipynb         # Model training & evaluation
├── src/
│   └── preprocessing/
│       ├── __init__.py
│       ├── missing_values.py
│       ├── data_cleaning.py
│       ├── feature_engineering.py
│       └── data_transformation.py
└── README.md
```

---

## Setup

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/fraud-detection.git
   cd fraud-detection
   ```

2. **Install dependencies:**
   ```sh
   pip install pandas scikit-learn numpy xgboost imbalanced-learn
   ```

3. **Prepare data:**
   - Place your raw data files in `data/raw/`:
     - `Fraud_Data.csv`
     - `IpAddress_to_Country.csv`

---

## Usage

### 1. Data Preprocessing

- Open `notebooks/fraud_notebook.ipynb`.
- Run all cells to:
  - Load and clean the data
  - Handle missing values
  - Engineer features (including IP address processing)
  - Save processed datasets to `data/processed/`

### 2. Model Training & Evaluation

- Open `notebooks/Model training.ipynb`.
- Run all cells to:
  - Load processed data
  - Train Logistic Regression, Random Forest, and XGBoost models
  - Evaluate models using F1 Score and AUC-PR
  - Print a summary of results for both datasets

---

## Customization

- Modify preprocessing steps in `src/preprocessing/` as needed.
- Add or tune models in `Model training.ipynb`.

---

## Notes

- The pipeline is modular: preprocessing and modeling are separated for clarity.
- All intermediate and final datasets are saved in `data/processed/`.
- Ensure the `src` directory is on your Python path (the notebooks handle this automatically).

