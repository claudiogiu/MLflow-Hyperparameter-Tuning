# Hyperparameter Tuning via MLflow 

## Introduction  

This repository is designed for hyperparameter tuning via MLflow. The optimization process is performed via GridSearchCV, ensuring the selection of the best-performing model. MLflow Tracking systematically logs parameters, metrics, and artifacts, enabling detailed monitoring of experiments. The top-performing model is then registered and versioned in MLflow Model Registry, laying the groundwork for a deployment strategy. The model is trained on a dataset sourced from the research conducted by KOKLU M., TASPINAR Y.S. (2021) in their paper *Determining the Extinguishing Status of Fuel Flames With Sound Wave by Machine Learning Methods* (IEEE Access, Vol. 9, DOI: [10.1109/ACCESS.2021.3088612](https://doi.org/10.1109/ACCESS.2021.3088612)).  

The dataset used for training is publicly available at [this link](https://www.muratkoklu.com/datasets/).  


## Getting Started 

To set up the repository properly, follow these steps:  

**1.** **Create the Data Directory**  
   - Before running the pipeline, create a `data/` folder in the project root.  
   - Inside `data/`, create two subdirectories:  
     - `raw/`: This will store the unprocessed dataset.  
     - `processed/`: The data will be split into **training and test sets** and saved here.
  
**2. Set Up the Conda Environment**  
 
   - Create and activate the **Conda** environment using the configuration in `conda.yaml`:  

     ```sh
     conda env create -f conda.yaml  
     conda activate mlflow_env  
     ```  

**3. Start the MLflow Server**  
 
   - Launch the **MLflow server** on `localhost:5000` with:  

     ```sh
     mlflow server --host 127.0.0.1 --port 5000
     ```    
   
   - When started, MLflow initializes key directories for managing experiment data:  
     - `mlruns/`: Created immediately to store logs, parameters, and metrics from each run.  
     - `mlartifacts/`: Generated dynamically when an experiment saves artifacts such as models or output files. 

**4. Execute the Pipeline with Makefile**  
   - The repository includes a **Makefile** to automate execution of scripts in the `src/` folder.  
   - Run the following command to execute the full workflow:  

     ```sh
     make run_all  
     ```  
   
   - This command sequentially runs the following components:
     - `load_data.py`: Ingests the data.
     - `preprocess.py`: Stores the trained One-Hot Encoder model in `models/` for reuse in inference.  
     - `tune_model.py`: Performs hyperparameter tuning using GridSearchCV, logging experiment details via **MLflow Tracking**.  
     - `train_model.py`: Trains the best-selected model, registers it in the **MLflow Model Registry** and stores it in `models/`.  
     - `evaluate_model.py`: Logs the performance metrics of the `"champion"` model.  


## License  

This project is licensed under the **MIT License**, which allows for open-source use, modification, and distribution with minimal restrictions. For more details, refer to the file included in this repository.  
