# Fraud Detection Pipeline

This repository contains a modular pipeline for detecting fraudulent transactions using Python and scikit-learn. The project is organized for clarity and reproducibility, with all data processing steps separated into the `src/preprocessing` package.

## Project Structure

```
fraud-detection/
├── data/
│   ├── raw/
│   │   ├── Fraud_Data.csv
│   │   └── IpAddress_to_Country.csv
│   └── processed/
├── notebooks/
│   └── fraud_notebook.ipynb
├── src/
│   └── preprocessing/
│       ├── __init__.py
│       ├── missing_values.py
│       ├── data_cleaning.py
│       ├── feature_engineering.py
│       └── data_transformation.py
└── README.md
```

## Setup

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/fraud-detection.git
   cd fraud-detection
   ```

2. **Install dependencies:**
   You can install the required Python packages using pip:
   ```sh
   pip install pandas scikit-learn numpy
   ```

3. **Data:**
   Place your raw data files in the `data/raw/` directory:
   - `Fraud_Data.csv`
   - `IpAddress_to_Country.csv`

## Usage

The main workflow is in the Jupyter notebook:  
`notebooks/fraud_notebook.ipynb`

This notebook:
- Loads raw data
- Handles missing values
- Cleans the data
- Performs feature engineering
- Transforms data (scaling, encoding, balancing, splitting)
- Saves processed datasets to `data/processed/`

To run the notebook:
1. Open it in JupyterLab or VS Code.
2. Run all cells.

## Customization

- To modify preprocessing steps, edit the corresponding modules in `src/preprocessing/`.
- The pipeline is modular; you can swap or extend steps as needed.

## Notes

- Ensure the `src` directory is on your Python path (the notebook handles this automatically).
- All intermediate and final datasets are saved in `data/processed/`.

