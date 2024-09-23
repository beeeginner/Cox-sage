# Cox-sage
The code for the article 'Cox-Sage: Enhancing Cox proportional hazards model with interpretable graph neural networks for cancer prognosis' under review in Briefings in Bioinformatics
## Data
**Due to GitHub's storage limitations, we have uploaded the dataset to Kaggle. Link: [Kaggle Dataset](https://www.kaggle.com/datasets/ridgiemo/processed-gene-and-clinical-data)**
Including TCGA datasets for 7 types of cancer: LUSC,STAD, LUAD,HNSC, ESCA, LIHC, COAD. Each dataset contains a processed protein-coding gene expression file named `gene_expression.csv`, the corresponding processed clinical data in `clinical.csv`, and the adjacency list `adj_list.pkl` of patients' similarity graph.
We do not provide the original clinical datasets because the handling of ordinal and numerical attributes, as well as the selection of clinical features, involves a significant amount of manual work. By providing the processed datasets and graph, users can directly execute the code, making it easier to reproduce the results in the paper.
## Code for Cancer Prognostic Model Training

The code for model training is relatively simple, consisting of only two files: `train_single_model.py` and `train_all.py`. We recommend using `train_all.py` because it can train multiple models at once. The usage is as follows:

```bash
python train_all.py --interpreter_path your_interpreter_path  # Set to the current interpreter path if left empty
                     --data_path the directory where your dataset is stored  # For example, it contains folders for LIHC, COAD
                     --save_path the directory where you want to store the trained model parameters
                     --log_path the path to the log file,  # Records the best performance of each dataset's cross-validation fold and the c-index score
```
**Training the model requires a GPU with over 36GB of VRAM. Download the dataset from the Kaggle link we provided. Then, create a `data` folder and copy the datasets you need for training into it. 
We recommend training two datasets at a time, as training more than that can lead to excessive storage space overhead.**


## Code for prognostic gene discovery
I am currently organizing the structure of this part of the code and will upload it later. **Executing this part of the code requires trained model parameters, and the inference overhead requires more than 10GB of GPU memory. However, to facilitate reproduction, I am considering providing a lightweight model by storing the results of matrix multiplication, which can reduce the inference overhead to a level that can be run on a personal PC.**
