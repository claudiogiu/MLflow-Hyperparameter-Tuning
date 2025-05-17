import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from load_data import load_data
import warnings

warnings.filterwarnings("ignore")


def split_data(X: pd.DataFrame, y: pd.DataFrame):
    """Split X and y into train/test (80-20) and return the DataFrames."""
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def one_hot_encoding_train(X_train: pd.DataFrame):
    """Apply One-Hot Encoding on categorical features in train set and save the encoder model."""
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    if not categorical_features:
        return X_train

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_encoded = encoder.fit_transform(X_train[categorical_features])

    new_columns = encoder.get_feature_names_out(categorical_features)
    X_train_encoded = pd.DataFrame(X_train_encoded, columns=new_columns, index=X_train.index)

    # Save One-Hot Encoder model
    encoder_path = os.path.join("models", "one_hot_encoder.pkl")
    os.makedirs("models", exist_ok=True)
    with open(encoder_path, "wb") as f:
        pickle.dump(encoder, f)

    # Replace categorical columns with encoded ones
    X_train = X_train.drop(columns=categorical_features).join(X_train_encoded)

    return X_train


def one_hot_encoding_test(X_test: pd.DataFrame):
    """Load the saved One-Hot Encoder and apply it to the test set."""
    encoder_path = os.path.join("models", "one_hot_encoder.pkl")

    if not os.path.exists(encoder_path):
        raise FileNotFoundError("Error: One-Hot Encoder model not found. Run training first.")

    try:
        with open(encoder_path, "rb") as f:
            encoder = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading One-Hot Encoder: {str(e)}")

    categorical_features = X_test.select_dtypes(include=["object"]).columns.tolist()
    if not categorical_features:
        return X_test

    X_test_encoded = encoder.transform(X_test[categorical_features])
    new_columns = encoder.get_feature_names_out(categorical_features)
    X_test_encoded = pd.DataFrame(X_test_encoded, columns=new_columns, index=X_test.index)

    # Replace categorical columns with encoded ones
    X_test = X_test.drop(columns=categorical_features).join(X_test_encoded)

    return X_test


def save_dataframe(df: pd.DataFrame, filename: str, folder_type: str):
    """Save a given DataFrame in the specified folder ('raw' or 'processed')"""
    if df is None or df.empty:
        raise ValueError(f"Error: DataFrame '{filename}' is empty or not found.")

    valid_folders = ["raw", "processed"]
    if folder_type not in valid_folders:
        raise ValueError(f"Invalid folder type '{folder_type}'. Choose from {valid_folders}.")

    folder_path = os.path.join("data", folder_type)
    os.makedirs(folder_path, exist_ok=True)

    df.to_excel(os.path.join(folder_path, f"{filename}.xlsx"), index=False)
    print(f"'{filename}.xlsx' successfully saved in '{folder_type}'. Shape: {df.shape}")


if __name__ == "__main__":
    dataset_name = "Acoustic_Extinguisher_Fire_Dataset.xlsx"
    df = load_data(dataset_name, "raw")

    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train = one_hot_encoding_train(X_train)
    X_test = one_hot_encoding_test(X_test)

    for name, dataset in zip(["X_train", "X_test", "y_train", "y_test"],
                         [X_train, X_test, y_train, y_test]):
        save_dataframe(dataset, name, "processed")