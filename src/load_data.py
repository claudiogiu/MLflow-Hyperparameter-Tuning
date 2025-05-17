import os
import pandas as pd

def load_data(dataset_name: str, data_type: str) -> pd.DataFrame:
    """
    Loads an Excel dataset.

    Parameters:
        dataset_name (str): Name of the Excel file (e.g., 'dataset.xlsx').
        data_type (str): Data type ('raw' or 'processed').

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """

    project_root = os.path.dirname(os.path.dirname(__file__)) 
    file_path = os.path.join(project_root, "data", data_type, dataset_name)

 
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist. Path: {file_path}")

    try:
        df = pd.read_excel(file_path)
        print(f"Dataset '{dataset_name}' successfully loaded. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None