# Cox-sage
The code for the article 'Cox-Sage: Enhancing Cox proportional hazards model with interpretable graph neural networks for cancer prognosis' under review in Briefings in Bioinformatics
## Data
Including TCGA datasets for 7 types of cancer: LUSC,STAD, LUAD,HNSC, ESCA, LIHC, COAD. Each dataset contains a processed protein-coding gene expression file named `expression.csv`, the corresponding processed clinical data in `clinical.csv`, and the adjacency list `adj_list.pkl` of patients' similarity graph.
├── data/
    ├── COAD/
        └── Adj_list.pkl
        └── clinical.csv
        └── gene_expression.csv
   ...
We do not provide the original clinical datasets because the handling of ordinal and numerical attributes, as well as the selection of clinical features, involves a significant amount of manual work. By providing the processed datasets and graph, users can directly execute the code, making it easier to reproduce the results in the paper.
## Code for Cancer Prognostic Model Training
The code for model training is relatively simple, consisting of only two files: `train_single_model.py` and `train_all.py`. We recommend using `train_all.py` because it can train multiple models at once. The usage is as follows:

python train_all.py --interpreter_path your_interpreter_path (set to the current interpreter path if left empty)
                     --data_path the directory where your dataset is stored (for example, it contains folders for LIHC, COAD)
                     --save_path the directory where you want to store the trained model parameters
                     --log_path the path to the log file, which records the best performance of each dataset's cross-validation fold and the c-index score

It is important to note that it is best not to train all seven datasets in `data` at once, as this will significantly increase the storage overhead and may fill up your storage space. It is recommended to choose two datasets at a time for training. For example, create a folder named `data_to_train` to pass as the `--data_path` argument, and copy the two datasets you want to train into it.
**Training the model requires a GPU with over 36GB of VRAM. If you are a reviewer for the journal and find that you do not have the necessary resources to run this code, please do not hesitate to contact us via email. We would be more than happy to arrange access to a server for you for a few hours to assist with your review. Thank you for your understanding and support.**


## Code for prognostic gene discovery
**I haven't organized the code for this part yet. I am considering compressing the model parameters by storing the results of matrix multiplication, which can significantly reduce the inference cost**
