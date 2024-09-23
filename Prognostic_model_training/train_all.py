import subprocess
import os
import argparse
import sys

if __name__ == '__main__':
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Training script parameters.')

    # Adding arguments with no default for interpreter_path
    parser.add_argument('--interpreter_path', type=str,
                        help='Path to the Python interpreter (leave empty to use current interpreter).')
    parser.add_argument('--data_path', type=str, default='data',
                        help='Path to the data directory.')
    parser.add_argument('--save_path', type=str, default='save',
                        help='Path to the directory where models will be saved.')
    parser.add_argument('--log_path', type=str, default='log.txt',
                        help='Path to the log file.')

    # Parse the arguments
    args = parser.parse_args()

    # Use the current interpreter path if not provided
    interpreter_path = args.interpreter_path if args.interpreter_path else sys.executable
    data_path = args.data_path
    save_path = args.save_path
    log_path = args.log_path

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Recording train log
    with open(log_path, 'a') as f:
        f.write('')

    cwd = os.getcwd()
    print(f'Current working directory: {cwd}')

    diseases = os.listdir(data_path)
    layers = [1, 2, 4]

    # Outer loop processes each disease in order
    for disease in diseases:
        print(f'Training prognostic model for {disease}...')
        # Inner loop executes different layers in parallel
        for l in layers:
            print(f'layers: {l}...')
            if disease not in ['LIHC','COAD']:
                subprocess.run([
                    interpreter_path, "train_single_model.py",
                    "--data_root", data_path,
                    "--model_save_root", save_path,
                    "--disease", disease,
                    "--num_layers", str(l),
                    "--log_path", log_path,
                ])
            elif disease=='LIHC':
                subprocess.run([
                    interpreter_path, "train_single_model.py",
                    "--data_root", data_path,
                    "--model_save_root", save_path,
                    "--disease", disease,
                    "--num_layers", str(l),
                    "--log_path", log_path,
                    "--seed",str(1),
                ])
            else:
                subprocess.run([
                    interpreter_path, "auto_train.py",
                    "--data_root", data_path,
                    "--model_save_root", save_path,
                    "--disease", disease,
                    "--num_layers", str(l),
                    "--log_path", log_path,
                    "--seed", str(9),
                ])








