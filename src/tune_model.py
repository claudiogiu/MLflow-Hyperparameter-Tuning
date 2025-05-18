import os
import json
import yaml
import mlflow
import mlflow.sklearn
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import mlflow.models 

warnings.filterwarnings("ignore")

def load_conda_env():
    """Load the Conda YAML file that describes the MLflow environment."""
    conda_env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "conda.yaml"))

    if not os.path.exists(conda_env_path):
        raise FileNotFoundError(f"Error: The configuration file '{conda_env_path}' does not exist.")

    try:
        with open(conda_env_path, "r") as file:
            conda_env = yaml.safe_load(file) 
    except yaml.YAMLError as e:
        raise ValueError(f"Error in decoding Conda YAML: {e}")

    return conda_env


def load_config():
    """Loads tuning parameters from an external config.json file outside the src folder."""
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.json"))

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Error: The configuration file '{config_path}' does not exist")

    try:
        with open(config_path, "r") as file:
            config_data = json.load(file)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error in decoding JSON: {e}")

    if "hyperparameters" not in config_data:
        raise KeyError("Error: The key 'hyperparameters' is not present in the configuration file.")

    return config_data["hyperparameters"]


def configure_mlflow(tracking_uri, experiment_name):
    """Configure MLflow with tracking URI and experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_run(gridsearch, run_index, model_name, tracking_uri, X_train, conda_env, tags={}):
    """Logs the results of hyperparameter tuning in MLflow."""
    mlflow.set_tracking_uri(tracking_uri) 

    cv_results = gridsearch.cv_results_
    with mlflow.start_run(run_name=f"Run_{run_index}", nested=True):
        mlflow.log_param("folds", gridsearch.cv)

        for param in gridsearch.param_grid.keys():
            mlflow.log_param(param, cv_results[f"param_{param}"][run_index])

        for score_name in [score for score in cv_results if "mean_test" in score]:
            mlflow.log_metric(score_name, cv_results[score_name][run_index])

            std_score_name = score_name.replace("mean", "std")
            mlflow.log_metric(std_score_name, cv_results[std_score_name][run_index])

        input_example = X_train.iloc[:1]  
        signature = mlflow.models.infer_signature(X_train, gridsearch.best_estimator_.predict(X_train))

        mlflow.sklearn.log_model(gridsearch.best_estimator_, model_name, 
                                 signature=signature, input_example=input_example, conda_env=conda_env)

        mlflow.set_tags(tags)

        print(f"Run {run_index} logged successfully: {mlflow.get_artifact_uri()}")


def tune_model(X, y, tracking_uri):
    """Performs hyperparameter tuning and logs each experiment in MLflow, identifying the best model."""
    configure_mlflow(tracking_uri, "Hyperparameter_Tuning")

    params = load_config()
    conda_env = load_conda_env()

    param_grid = {
        "n_estimators": params["n_estimators"],
        "criterion": params["criterion"],
        "max_depth": params["max_depth"],
        "max_features": params["max_features"],
        "bootstrap": params["bootstrap"]
    }

    rf = RandomForestClassifier(random_state=params["random_state"][0])

    grid_search = GridSearchCV(
        rf, 
        param_grid, 
        cv=params["cv"], 
        scoring=params["scoring"], 
        n_jobs=params["n_jobs"], 
        return_train_score=True
    )
    grid_search.fit(X, y)

    # Logging of the experiments
    num_runs = len(grid_search.cv_results_["mean_test_score"])
    for run_index in range(num_runs):
        log_run(grid_search, run_index, "RandomForest_Experiment", tracking_uri, X, conda_env)

    # Logging of the best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_std_score = grid_search.cv_results_["std_test_score"][grid_search.best_index_]

    with mlflow.start_run(run_name="Best_Model"):
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", best_score)
        mlflow.log_metric("accuracy_std", best_std_score)

        input_example = X.iloc[:1]  
        signature = mlflow.models.infer_signature(X, best_model.predict(X))

        mlflow.sklearn.log_model(best_model, "best_model", signature=signature, input_example=input_example)

    print(f"Best model found with accuracy {best_score * 100:.2f} ± {best_std_score * 100:.2f} (%) and parameters {best_params}.")


if __name__ == "__main__":
    tracking_uri = "http://localhost:5000"

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    X_path = os.path.join(project_root, "data", "processed", "X_train.xlsx")
    y_path = os.path.join(project_root, "data", "processed", "y_train.xlsx")

    X = pd.read_excel(X_path)
    y = pd.read_excel(y_path).values.ravel()

    tune_model(X, y, tracking_uri)
