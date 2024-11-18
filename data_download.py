import argparse
import kagglehub
import os
import shutil

parser = argparse.ArgumentParser(description="Download and copy a Kaggle dataset.")
parser.add_argument(
    "--target",
    type=str,
    default="dataset",
    help="Target directory to copy the dataset files. Default is 'dataset'."
)

args = parser.parse_args()
current_dir = os.getcwd()

path = kagglehub.dataset_download("ridgiemo/processed-gene-and-clinical-data")
print("Dataset Cache Path:", path)

target_directory = os.path.join(current_dir, args.target)
if os.path.exists(target_directory):
    print(f"Target directory '{target_directory}' already exists. We will merge the contents.")
else:
    os.makedirs(target_directory)

for item in os.listdir(path):
    source_item = os.path.join(path, item)
    destination_item = os.path.join(target_directory, item)

    if os.path.isdir(source_item):
        shutil.copytree(source_item, destination_item)  
    else:
        shutil.copy2(source_item, destination_item)  

print(f"All files have been copied to: {target_directory}")


