import subprocess
import argparse
import sys
import os

def download_dataset(target_directory=None):

    command = [sys.executable, 'data_download.py']  

    if target_directory:
        command.append('--target')
        command.append(target_directory)
    try:
        result = subprocess.run(command)
        print("Output:\n", result.stdout)
        print("Errors:\n", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing the script: {e}")
        print("Output:\n", e.stdout)
        print("Errors:\n", e.stderr)
def train_models(data_path,model_save_path,disease_name):

    command = [sys.executable, 'train_all_models.py',
               '--data_path',data_path,
               '--save_path',model_save_path,
               '--disease',disease_name]
    try:
        result = subprocess.run(command)
        print("Output:\n", result.stdout)
        print("Errors:\n", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing the script: {e}")
        print("Output:\n", e.stdout)
        print("Errors:\n", e.stderr)
def inference_km(disease_name,dataset_root, num_layers,model_root,prediction_output_dir):
    command = [sys.executable, 'inference_km.py',
               '--disease_name', disease_name,
               '--dataset_root', dataset_root,
               '--num_layers', num_layers,
               '--model_root', model_root,
               '--prediction_output_dir', prediction_output_dir]
    try:
        result = subprocess.run(command)
        print("Output:\n", result.stdout)
        print("Errors:\n", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing the script: {e}")
        print("Output:\n", e.stdout)
        print("Errors:\n", e.stderr)
def prognostic_gene_discovery(mode,data_root,model_save_root,dataset_name,output_graph_root,prediction_save_path):
    command = [sys.executable, 'prognostic_gene_discovery.py',
               '--mode', mode,
               '--data_root', data_root,
               '--model_save_root', model_save_root,
               '--dataset_name', dataset_name,
               '--output_graph_root', output_graph_root,
               '--prediction_save_path',prediction_save_path]
    try:
        result = subprocess.run(command)
        print("Output:\n", result.stdout)
        print("Errors:\n", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing the script: {e}")
        print("Output:\n", e.stdout)
        print("Errors:\n", e.stderr)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train and reproduce in a single script.")
    parser.add_argument(
        "--data_storage_dir",
        type=str,
        default="dataset",
        help="Target directory name to storage dataset. Default is 'dataset'."
    )
    parser.add_argument(
        "--models_save_path",
        type=str,
        default="model_save",
        help="Target directory name to storage models. Default is 'model_save'."
    )
    parser.add_argument(
        "--disease_name",
        type=str,
        default="LIHC",
        help="The disease you want to research. Default is 'LIHC'."
    )
    parser.add_argument(
        "--prediction_save_path",
        type=str,
        default="prediction_output",
        help="The direction you want to storage risk prediction and km survival analysis"
    )
    parser.add_argument(
        "--prognostic_gene_path",
        type=str,
        default="prognostic_genes",
        help="The direction you want to storage prognostic genes discovery results."
    )
    args = parser.parse_args()
    if not os.path.exists(args.data_storage_dir):
        print(f'Downloading dataset......')
        download_dataset(args.data_storage_dir)
    if not os.path.exists(args.models_save_path):
        print(f'Training models......')
        dataset_path = os.path.join(args.data_storage_dir,'data')
        train_models(dataset_path,args.models_save_path,args.disease_name)
    if not os.path.exists(args.prediction_save_path):
        os.mkdir(args.prediction_save_path)
        print(f'Inference predictions and conducting KM survival analysis......')
        for num_layers in [1,2,4]:
            inference_km(disease_name=args.disease_name,
                         dataset_root=os.path.join(args.data_storage_dir,"data"),
                         num_layers=str(num_layers),
                         model_root=args.models_save_path,
                         prediction_output_dir=args.prediction_save_path)
    if not os.path.exists(args.prognostic_gene_path):
        print(f'Identifying genes whose low expression correlates with high risk...')
        prognostic_gene_discovery('MHZ', os.path.join(args.data_storage_dir,'data'), args.models_save_path, args.disease_name, args.prognostic_gene_path,args.prediction_save_path)
        print(f'Identifying genes whose high expression correlates with high risk...')
        prognostic_gene_discovery('RMHZ', os.path.join(args.data_storage_dir,'data'), args.models_save_path, args.disease_name, args.prognostic_gene_path, args.prediction_save_path)



    
