import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_data(path, target_column):
    df = pd.read_csv(path)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def train_best_model(X_train, y_train):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    return model

def run_shap_explainer(model, X_sample):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)

    print("✅ SHAP values calculated.")

    # Global feature importance
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig("outputs/shap_summary_plot.png")
    print("📊 Summary plot saved to outputs/shap_summary_plot.png")

    # Force plot for a single prediction
    force_plot = shap.plots.force(shap_values[0], matplotlib=True)
    plt.title("SHAP Force Plot (Local Explanation)")
    plt.savefig("outputs/shap_force_plot_sample0.png")
    print("🔍 Force plot saved to outputs/shap_force_plot_sample0.png")

    return shap_values

if __name__ == "__main__":
    # Load preprocessed dataset
    data_path = "data/processed/fraud_data_processed.csv"
    target_column = "class"
    X, y = load_data(data_path, target_column)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Train the best model (XGBoost)
    best_model = train_best_model(X_train, y_train)

    # Sample from test set for SHAP (optional: use full X_test if small)
    sample = X_test.sample(n=100, random_state=42)

    # Run SHAP explainability
    run_shap_explainer(best_model, sample)
