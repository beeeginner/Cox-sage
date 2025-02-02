import subprocess
import argparse
import sys
import os

# Function to download the dataset
def download_dataset(target_directory=None):
    # Prepare the command to run the data_download.py script
    command = [sys.executable, 'data_download.py']

    # If a target directory is specified, add it to the command
    if target_directory:
        command.append('--target')
        command.append(target_directory)
    
    try:
        # Run the download script as a subprocess and capture output
        result = subprocess.run(command)
        print("Output:\n", result.stdout)
        print("Errors:\n", result.stderr)
    except subprocess.CalledProcessError as e:
        # If an error occurs during the subprocess execution, print the error
        print(f"An error occurred while executing the script: {e}")
        print("Output:\n", e.stdout)
        print("Errors:\n", e.stderr)

# Function to train models using the specified dataset and disease
def train_models(data_path, model_save_path, disease_name):
    # Prepare the command to run the train_all_models.py script
    command = [sys.executable, 'train_all_models.py',
               '--data_path', data_path,
               '--save_path', model_save_path,
               '--disease', disease_name]
    
    try:
        # Run the training script as a subprocess and capture output
        result = subprocess.run(command)
        print("Output:\n", result.stdout)
        print("Errors:\n", result.stderr)
    except subprocess.CalledProcessError as e:
        # If an error occurs during the subprocess execution, print the error
        print(f"An error occurred while executing the script: {e}")
        print("Output:\n", e.stdout)
        print("Errors:\n", e.stderr)

# Function to run inference and KM survival analysis for given disease and models
def inference_km(disease_name, dataset_root, num_layers, model_root, prediction_output_dir):
    # Prepare the command to run the inference_km.py script
    command = [sys.executable, 'inference_km.py',
               '--disease_name', disease_name,
               '--dataset_root', dataset_root,
               '--num_layers', num_layers,
               '--model_root', model_root,
               '--prediction_output_dir', prediction_output_dir]
    
    try:
        # Run the inference script as a subprocess and capture output
        result = subprocess.run(command)
        print("Output:\n", result.stdout)
        print("Errors:\n", result.stderr)
    except subprocess.CalledProcessError as e:
        # If an error occurs during the subprocess execution, print the error
        print(f"An error occurred while executing the script: {e}")
        print("Output:\n", e.stdout)
        print("Errors:\n", e.stderr)

# Function to discover prognostic genes for given disease
def prognostic_gene_discovery(mode, data_root, model_save_root, dataset_name, output_graph_root, prediction_save_path):
    # Prepare the command to run the prognostic_gene_discovery.py script
    command = [sys.executable, 'prognostic_gene_discovery.py',
               '--mode', mode,
               '--data_root', data_root,
               '--model_save_root', model_save_root,
               '--dataset_name', dataset_name,
               '--output_graph_root', output_graph_root,
               '--prediction_save_path', prediction_save_path]
    
    try:
        # Run the prognostic gene discovery script as a subprocess and capture output
        result = subprocess.run(command)
        print("Output:\n", result.stdout)
        print("Errors:\n", result.stderr)
    except subprocess.CalledProcessError as e:
        # If an error occurs during the subprocess execution, print the error
        print(f"An error occurred while executing the script: {e}")
        print("Output:\n", e.stdout)
        print("Errors:\n", e.stderr)

# Main function to control the flow of the script
if __name__ == "__main__":

    # Argument parser to handle command line arguments
    parser = argparse.ArgumentParser(description="Train and reproduce in a single script.")
    
    # Argument to specify the directory to store the downloaded dataset
    parser.add_argument(
        "--data_storage_dir",
        type=str,
        default="dataset",
        help="Target directory name to store dataset. Default is 'dataset'."
    )
    
    # Argument to specify the directory to save trained models
    parser.add_argument(
        "--models_save_path",
        type=str,
        default="model_save",
        help="Target directory name to store models. Default is 'model_save'."
    )
    
    # Argument to specify the disease to research (e.g., LIHC)
    parser.add_argument(
        "--disease_name",
        type=str,
        default="LIHC",
        help="The disease you want to research. Default is 'LIHC'."
    )
    
    # Argument to specify where to save prediction and KM survival analysis output
    parser.add_argument(
        "--prediction_save_path",
        type=str,
        default="prediction_output",
        help="The directory where you want to store risk prediction and KM survival analysis results."
    )
    
    # Argument to specify where to save prognostic gene discovery results
    parser.add_argument(
        "--prognostic_gene_path",
        type=str,
        default="prognostic_genes",
        help="The directory where you want to store prognostic gene discovery results."
    )
    
    # Parse the arguments
    args = parser.parse_args()

    # Check if the data directory exists, if not, download the dataset
    if not os.path.exists(args.data_storage_dir):
        print(f'Downloading dataset...')
        download_dataset(args.data_storage_dir)
    
    # Check if the models directory exists, if not, train the models
    if not os.path.exists(args.models_save_path):
        print(f'Training models...')
        dataset_path = os.path.join(args.data_storage_dir, 'data')
        train_models(dataset_path, args.models_save_path, args.disease_name)
    
    # Check if the prediction directory exists, if not, run inference and KM analysis
    if not os.path.exists(args.prediction_save_path):
        os.mkdir(args.prediction_save_path)
        print(f'Inference predictions and conducting KM survival analysis...')
        for num_layers in [1, 2, 4]:  # Loop through different numbers of layers for the model
            inference_km(disease_name=args.disease_name,
                         dataset_root=os.path.join(args.data_storage_dir, "data"),
                         num_layers=str(num_layers),
                         model_root=args.models_save_path,
                         prediction_output_dir=args.prediction_save_path)
    
    # Check if the prognostic gene directory exists, if not, run the prognostic gene discovery
    if not os.path.exists(args.prognostic_gene_path):
        print(f'Identifying genes whose low expression correlates with high risk...')
        prognostic_gene_discovery('MHZ', os.path.join(args.data_storage_dir, 'data'),
                                   args.models_save_path, args.disease_name,
                                   args.prognostic_gene_path, args.prediction_save_path)
        print(f'Identifying genes whose high expression correlates with high risk...')
        prognostic_gene_discovery('RMHZ', os.path.join(args.data_storage_dir, 'data'),
                                   args.models_save_path, args.disease_name,
                                   args.prognostic_gene_path, args.prediction_save_path)




