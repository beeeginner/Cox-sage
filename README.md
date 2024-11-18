# Cox-sage
The code for the article 'Cox-Sage: Enhancing Cox proportional hazards model with interpretable graph neural networks for cancer prognosis' under review in Briefings in Bioinformatics
# Dataset Availability

We will upload the processed data to both Kaggle and Zendo. The Kaggle link is [here](https://www.kaggle.com/datasets/ridgiemo/processed-gene-and-clinical-data), and the Zendo link is [here](url2).

# Requirements

To set up the project, you will need to install the following Python packages. You can install them using `pip` with the provided `requirements.txt` file.

## Required Packages

```plaintext
kagglehub==0.3.4
lifelines==0.29.0
matplotlib==3.4.3
numpy==1.22.4
pandas==2.2.3
scikit_learn==0.24.2
torch==2.0.0
torch_geometric==2.6.1
```

## Installation

To install the required packages, run the following command in your terminal:

```bash
pip install -r requirements.txt
```

# Run code and reproduce all the results of the paper
After installing the requirements and downloading this repository, you can reproduce all the results from our paper with just one line of code:

```bash
python run.py --disease <Disease Name>
```

This command will automatically execute the following workflow:
- Dataset download
- Training on the specified dataset (default is LIHC)
- Outputting prediction results
- Performing Kaplan-Meier survival analysis based on the output results and plotting the curves
- Discovering cancer prognosis genes and generating visualizations


