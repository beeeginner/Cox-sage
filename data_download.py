'''
    Download 7 processed TCGA datasets.
'''
import argparse  # For handling command-line arguments
import kagglehub  # Library to download Kaggle datasets
import os  # Provides operating system interfaces, such as file path operations
import shutil  # For file and directory copying

# Set up argument parser for the script
parser = argparse.ArgumentParser(description="Download and copy a Kaggle dataset.")
parser.add_argument(
    "--target",  # Argument to specify the target directory
    type=str,  # Argument type is string
    default="dataset",  # Default directory name if not specified
    help="Target directory to copy the dataset files. Default is 'dataset'."
)

# Parse the command-line arguments
args = parser.parse_args()

# Get the current working directory
current_dir = os.getcwd()

# Download the Kaggle dataset and get the cached path
path = kagglehub.dataset_download("ridgiemo/processed-gene-and-clinical-data")
print("Dataset Cache Path:", path)

# Construct the full path of the target directory
target_directory = os.path.join(current_dir, args.target)

# Check if the target directory already exists
if os.path.exists(target_directory):
    print(f"Target directory '{target_directory}' already exists. We will merge the contents.")
else:
    # Create the directory if it doesn't exist
    os.makedirs(target_directory)

# Iterate through the downloaded dataset and copy files/directories
for item in os.listdir(path):
    source_item = os.path.join(path, item)
    destination_item = os.path.join(target_directory, item)

    # If the item is a directory, use copytree to copy recursively
    if os.path.isdir(source_item):
        shutil.copytree(source_item, destination_item)  
    else:
        # If the item is a file, use copy2 to preserve metadata
        shutil.copy2(source_item, destination_item)  

# Notify the user that all files have been copied successfully
print(f"All files have been copied to: {target_directory}")



