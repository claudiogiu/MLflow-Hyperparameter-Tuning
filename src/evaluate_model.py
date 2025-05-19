import os
import pickle
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from load_data import load_data
import warnings

warnings.filterwarnings("ignore")


def log_model_metrics(model_name: str, accuracy: float, precision: float, sensitivity: float, 
                         specificity: float, auc: float, f1: float, tracking_uri: str):
    """Log validation metrics for the model in MLflow."""
    
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()


    versions = client.get_latest_versions(model_name)
    if not versions:
        raise ValueError(f"No registered model found with name: {model_name}")


    run_id = versions[0].run_id
    client.log_metric(run_id=run_id, key="accuracy", value=accuracy)
    client.log_metric(run_id=run_id, key="precision", value=precision)
    client.log_metric(run_id=run_id, key="sensitivity", value=sensitivity)
    client.log_metric(run_id=run_id, key="specificity", value=specificity)
    client.log_metric(run_id=run_id, key="auc", value=auc)
    client.log_metric(run_id=run_id, key="f1_score", value=f1)


def test_model(X_test: pd.DataFrame, y_test: pd.DataFrame):
    """Load the saved model and evaluate its performance."""
    
    models_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

    # Load the model
    champion_model_path = os.path.join(models_path, "champion_model.pkl")
    if not os.path.exists(champion_model_path):
        raise FileNotFoundError(f"Model not found. Expected at: {champion_model_path}")

    with open(champion_model_path, "rb") as f:
        champion_model = pickle.load(f)

    # Predict using the model
    y_pred = champion_model.predict(X_test)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred) * 100
    sensitivity = recall_score(y_test, y_pred, pos_label=1, average="binary") * 100
    specificity = recall_score(y_test, y_pred, pos_label=0, average="binary") * 100
    auc = roc_auc_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100

    print("Model Evaluation Metrics:")
    print(f"Accuracy Score (%): {accuracy:.2f}")
    print(f"Precision Score (%): {precision:.2f}")
    print(f"Sensitivity Score (%): {sensitivity:.2f}")
    print(f"Specificity Score (%): {specificity:.2f}")
    print(f"AUC Score (%): {auc:.2f}") 
    print(f"F1 Score (%): {f1:.2f}") 

    log_model_metrics("Champion_Model", accuracy, precision, sensitivity, specificity, auc, f1, tracking_uri)

if __name__ == "__main__":
    tracking_uri = "http://localhost:5000"
    
    X_test = load_data("X_test.xlsx", "processed")
    y_test = load_data("y_test.xlsx", "processed")

    test_model(X_test, y_test)