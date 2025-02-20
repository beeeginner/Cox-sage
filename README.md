# Cox-sage
The code for the article 'Cox-Sage: Enhancing Cox proportional hazards model with interpretable graph neural networks for cancer prognosis' under review in Briefings in Bioinformatics
# Dataset Availability

We will upload the processed data to both Kaggle and Zendo. The Kaggle link is [https://www.kaggle.com/datasets/ridgiemo/processed-gene-and-clinical-data](https://www.kaggle.com/datasets/ridgiemo/processed-gene-and-clinical-data), and the Zendo link is [https://zenodo.org/records/14204893](https://zenodo.org/records/14204893).You can also find the prediction outputs in the latest version of the zendo link.

# Experimental Results in the Paper
To reproduce Table 3, run the code at: [https://github.com/beeeginner/benchmarks_compare](https://github.com/beeeginner/benchmarks_compare). This link includes the code and prediction outputs of all reproduced benchmark methods, as well as the repeated experiments of the Cox-sage method under different random seeds. To reproduce Figures 2 and 3, you can follow the steps below to replicate the code in this repository.

# Requirements

To set up the project, you will need to install the following Python packages. You can install them using `pip` with the provided `requirements.txt` file.

## Required Packages

```plaintext
kagglehub==0.3.7
lifelines==0.27.8
matplotlib==3.7.5
numpy==1.24.4
pandas==2.0.3
scikit_learn==1.3.2
torch==2.0.0+cu118
torch_geometric==2.6.1
```

## Installation

To install the required packages, run the following command in your terminal:

```bash
pip install -r requirements.txt
```

# Run code and reproduce all the results of the paper
After installing the requirements and downloading this repository (Just download all the code; there is no need to download the folder. The result folder will be automatically generated at the end of the code execution.), you can reproduce all the results from our paper with just one line of code:

```bash
python run.py --disease <Disease Name>
```

This command will automatically execute the following workflow:
- Dataset download
- Training on the specified dataset (default is LIHC)
- Outputting prediction results
- Performing Kaplan-Meier survival analysis based on the output results and plotting the curves
- Discovering cancer prognosis genes and generating visualizations

### Optional Parameters

- `--disease` (str): 
  - **Description**: The abbreviation of the cancer dataset you want to train, such as LIHC, LUSC.
  - **Default**: LIHC

- `--models_save_path` (str): 
  - **Description**: Target directory name to storage models.
  - **Default**: 'model_save'(If a folder with the same name appears in the working directory, the model training will be skipped.)

- `--prediction_save_path` (str): 
  - **Description**: The direction you want to storage risk prediction and km survival analysis.
  - **Default**: 'prediction_output'(If a folder with the same name appears in the working directory,  the model's inference and KM survival analysis steps will be skipped.)
 
- `--prognostic_gene_path` (str): 
  - **Description**: The direction you want to storage prognostic genes discovery results.
  - **Default**: 'prediction_output'(If a folder with the same name appears in the working directory, the model-based prognostic genes discovery step will be skipped.)
    
# Script Explanation

The following is an explanation of each script and its functionality:

- **`data_download.py`**  
  Automatically downloads the dataset from the Kaggle link and extracts it to the current directory.

- **`train_single_model.py`**  
  Trains a single model on the specified disease.

- **`train_all_models.py`**  
  Trains all models for all diseases.

- **`inference_km.py`**  
  Loads the specified trained model on the dataset of the specified disease, performs Kaplan-Meier survival analysis based on its predicted outputs, and plots the survival curve.

- **`prognostic_gene_discovery.py`**  
  Identifies prognostic genes for the specified disease using the trained Cox-Sage model.

- **`run.py`**  
  Executes all the processes for the specified disease.



