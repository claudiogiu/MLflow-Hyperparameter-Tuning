import os
import pickle
from load_data import load_data
import mlflow
import mlflow.sklearn
from tune_model import configure_mlflow
import warnings

warnings.filterwarnings("ignore")

def load_best_model(experiment_name, tracking_uri):
    """Finds the best model from the specified MLflow experiment."""
    configure_mlflow(tracking_uri, experiment_name) 

    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    best_run = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.accuracy DESC"], max_results=1)

    if not best_run:
        raise ValueError("No best model found in MLflow experiment.")

    best_model_uri = best_run[0].info.artifact_uri + "/best_model"
    return mlflow.sklearn.load_model(best_model_uri)


def retrain_model(model, X, y):
    """Retrains the given model on the full dataset."""
    print("Retraining the best model on the training set...")
    model.fit(X, y)
    return model


def register_model(model, tracking_uri, X, alias):
    """Register the retrained model in the MLflow Model Registry."""
    mlflow.set_tracking_uri(tracking_uri)

    with mlflow.start_run(run_name="Champion_Model"):
        X_sample = X.iloc[:1]  
        mlflow.sklearn.log_model(model, "champion_model", registered_model_name="Champion_Model", input_example=X_sample)
    
    client = mlflow.MlflowClient()
    
    versions = [v.version for v in client.get_latest_versions("Champion_Model")]
    latest_version = max(versions) if versions else 1
    
    client.set_registered_model_alias(name="Champion_Model", version=latest_version, alias=alias)

    print(f"Model successfully registered with alias '{alias}'.")


def save_model(model):
    """Saves the retrained model in '/models'."""
    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "champion_model.pkl")
    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    print(f"Model saved at: {model_path}")


if __name__ == "__main__":
    tracking_uri = "http://localhost:5000"
    experiment_name = "Hyperparameter_Tuning"

    X = load_data("X_train.xlsx", "processed")
    y = load_data("y_train.xlsx", "processed").values.ravel()

    best_model = load_best_model(experiment_name, tracking_uri)
    retrained_model = retrain_model(best_model, X, y)
    register_model(retrained_model, tracking_uri, X, alias="champion")
    save_model(retrained_model)
