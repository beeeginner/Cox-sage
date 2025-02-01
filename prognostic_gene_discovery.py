import pandas as pd  # Data manipulation library
import torch  # PyTorch library for deep learning
from torch.nn import init  # Initialization functions for neural networks
from torch_geometric.nn import SAGEConv  # GraphSAGE convolutional layer from PyTorch Geometric
import torch.nn as nn  # Neural network module from PyTorch
import numpy as np  # Numerical operations library
import pickle as pkl  # Library for serializing and deserializing Python objects
import argparse  # Library for parsing command-line arguments
import matplotlib.pyplot as plt  # Plotting library
import os  # Operating system functions like file and directory management

# Create an argument parser to handle command-line inputs
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, help="MHZ OR RMHZ")  # Mode for hazard prediction
parser.add_argument('--data_root', type=str, help="Root directory of the dataset", default="data")  # Directory of dataset
parser.add_argument('--model_save_root', type=str, help="Root that you save trained models", default="save")  # Model save location
parser.add_argument('--dataset_name', type=str, help="The dataset you want to conduct prognostic gene discovery on.", default="LIHC")  # Dataset name
parser.add_argument("--prediction_save_path", type=str, default="prediction_output", help="Directory to store risk prediction and survival analysis")  # Path for saving prediction results
parser.add_argument('--output_graph_root', type=str, help="Root that you want to save the graph.", default='graphs')  # Graph output directory

# Parse the arguments provided via the command-line
args = parser.parse_args()
mode = args.mode  # Mode (either MHZ or RMHZ)
data_root = args.data_root  # Root path for the dataset
dataset_name = args.dataset_name  # Dataset name (e.g., LIHC)
model_save_root = args.model_save_root  # Root path for saving models
graph_root = args.output_graph_root  # Root path for saving generated graphs

# Create directories for saving graphs if they don't already exist
if not os.path.exists(graph_root):
    os.mkdir(graph_root)
    os.mkdir(os.path.join(graph_root, 'low_expression_high_hazards'))
    os.mkdir(os.path.join(graph_root, 'high_expression_high_hazards'))

# Ensure that the 'mode' argument is either 'MHZ' or 'RMHZ'
assert mode in ('MHZ', 'RMHZ'), r'Mode must be set to MHZ or RMHZ'

# Determine whether to use GPU or CPU for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the gene expression data from a CSV file and sort by patient ID
df = pd.read_csv(f'{data_root}/{dataset_name}/gene_expression.csv', header=0)
df = df.sort_values(by='patient_id')
df = df.reset_index(drop=True)

# Load the adjacency list which represents the relationships between genes
with open(f'{data_root}/{dataset_name}/Adj_list.pkl', 'rb') as f:
    neibors = pkl.load(f)

# Remove patient_id column and store gene names
del df['patient_id']
gene_names = df.columns.values.tolist()

# Convert gene expression values into a numpy array and apply log2 transformation
X = df.values
X = np.log2(X + 1)

# Define the SageCox model using the GraphSAGE layers
class SageCox(nn.Module):
    def __init__(self, num_layers=4):
        super(SageCox, self).__init__()
        in_channels = X.shape[1]  # Number of input features (genes)
        out_channels = 1  # Output dimension (risk score)
        self.num_layers = num_layers  # Number of GraphSAGE layers

        # Define layers based on the specified number of layers
        if num_layers == 4:
            n1 = int(in_channels * (2 / 3))
            n2 = int(n1 * (2 / 3))
            n3 = int(n2 / 2)
            self.GNN1 = SAGEConv(in_channels=in_channels, out_channels=n1)
            self.GNN2 = SAGEConv(in_channels=n1, out_channels=n2)
            self.GNN3 = SAGEConv(in_channels=n2, out_channels=n3)
            self.GNN4 = SAGEConv(in_channels=n3, out_channels=out_channels)
        elif num_layers == 2:
            hidden_channels = int(in_channels / 2)
            self.GNN1 = SAGEConv(in_channels=in_channels, out_channels=hidden_channels)
            self.GNN2 = SAGEConv(in_channels=hidden_channels, out_channels=out_channels)
        elif num_layers == 1:
            self.GNN1 = SAGEConv(in_channels=in_channels, out_channels=out_channels)
        else:
            raise Exception("Invalid number of layers.")

    def forward(self, data):
        # Forward pass through the GraphSAGE layers
        x, edge_index, h = data.x, data.edge_index, data.h
        if self.num_layers == 4:
            output = self.GNN1(x=x, edge_index=edge_index)
            output = self.GNN2(x=output, edge_index=edge_index)
            output = self.GNN3(x=output, edge_index=edge_index)
            output = self.GNN4(x=output, edge_index=edge_index)
        elif self.num_layers == 2:
            output = self.GNN1(x=x, edge_index=edge_index)
            output = self.GNN2(x=output, edge_index=edge_index)
        else:
            output = self.GNN1(x=x, edge_index=edge_index)
        
        # If a flag (not defined) is set, reshape output
        if self._flag == 1:
            output = output.view(-1)
        return output

# Calculate the average gene expression of neighbors for each sample
Y = np.zeros_like(X)
for v in range(X.shape[0]):
    neibor = neibors[v]
    Y[v] = np.mean(X[neibor], axis=0)

# Initialize lists to store the model parameters
A, B = [], []
layers = [1, 2, 4]

# Load pre-trained models with different layers and extract parameters
for l in layers:
    model = SageCox(num_layers=l).to(device)
    checkpoint = torch.load(f'{model_save_root}/{dataset_name}/{l}/best_fold.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model_params = model.state_dict()

    if l == 1:
        with torch.no_grad():
            alpha, beta = model_params['GNN1.lin_l.weight'].cpu().numpy(), model_params['GNN1.lin_l.weight'].cpu().numpy()
        A.append(alpha.squeeze())
        B.append(beta.squeeze())
    elif l == 2:
        with torch.no_grad():
            W1, W2 = model_params['GNN1.lin_l.weight'], model_params['GNN1.lin_r.weight']
            alpha, beta = model_params['GNN2.lin_l.weight'], model_params['GNN2.lin_l.weight']
            a = W1.T @ alpha.T
            b = W2.T @ beta.T
            del W1, W2, alpha, beta
            A.append(a.cpu().numpy().squeeze())
            B.append(b.cpu().numpy().squeeze())
    elif l == 4:
        with torch.no_grad():
            W1, W2 = model_params['GNN1.lin_l.weight'], model_params['GNN1.lin_r.weight']
            W3, W4 = model_params['GNN2.lin_l.weight'], model_params['GNN2.lin_r.weight']
            W5, W6 = model_params['GNN3.lin_l.weight'], model_params['GNN3.lin_r.weight']
            alpha, beta = model_params['GNN4.lin_l.weight'], model_params['GNN4.lin_l.weight']
            a = W1.T @ W3.T @ W5.T @ alpha.T
            del W1, W3, W5, alpha
            b = W2.T @ W4.T @ W6.T @ beta.T
            del W2, W4, W6, beta
            A.append(a.cpu().numpy().squeeze())
            B.append(b.cpu().numpy().squeeze())

# Identify genes selected across all layers
all_selected_genes = []
for idx, l in enumerate([1, 2, 4]):
    a, b = A[idx], B[idx]
    delta_thetas = []
    for gene in range(X.shape[1]):
        expression_vector = X[:, gene].copy()
        expression_y = Y[:, gene].copy()
        delta_thetas.append(-np.mean(a[gene] * expression_vector + b[gene] * expression_y))

    delta_thetas = np.array(delta_thetas)
    if mode == 'MHZ':
        mhzs = delta_thetas.tolist()
        df = pd.DataFrame({'gene_name': gene_names, 'score': mhzs})
    else:
        rmhzs = -delta_thetas
        df = pd.DataFrame({'gene_name': gene_names, 'score': rmhzs})

    # Filter genes based on score thresholds
    scs = df['score'].values
    q3 = np.quantile(scs, q=0.75)
    q1 = np.quantile(scs, q=0.25)
    IQR = q3 - q1
    mid = np.quantile(scs, q=0.5)
    lb = mid
    df = df[df['score'] > lb].reset_index(drop=True)
    select = set(df['gene_name'].values.tolist())
    all_selected_genes.append(select)

# Find genes that are selected across all layers
print(f'1layer {len(all_selected_genes[0])} 2layers {len(all_selected_genes[1])} 4layers {len(all_selected_genes[2])}')
selected_genes = all_selected_genes[0] & all_selected_genes[1] & all_selected_genes[2]
selected_genes = list(selected_genes)

print(f'{mode} selecting {len(selected_genes)} genes in total')

# Save selected genes to files
if mode == 'MHZ':
    with open(f'{os.path.join(graph_root,"low_expression_high_hazards")}/gene_discovery.txt', 'a') as f:
        f.write(f'genes that low expression leading to high hazards: {selected_genes}\n')
    with open(f'{os.path.join(graph_root,"low_expression_high_hazards")}/gene_discovery.pkl', 'wb') as f:
        pkl.dump(selected_genes, f)
else:
    with open(f'{os.path.join(graph_root,"high_expression_high_hazards")}/gene_discovery.txt', 'a') as f:
        f.write(f'genes that high expression leading to high hazards: {selected_genes}\n')
    with open(f'{os.path.join(graph_root,"high_expression_high_hazards")}/gene_discovery.pkl', 'wb') as f:
        pkl.dump(selected_genes, f)

# Define a binary function for the model output
class binary_function:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def f(self, x, y):
        return np.exp(self.a * x + self.b * y)

# Function to plot the contour map of hazards
def plot_contour_map(ax, a, b, xlow, ylow, xhigh, yhigh, l, L, gene_name, name):
    function = binary_function(a, b)
    x = np.linspace(l, L, 100)
    y = np.linspace(l, L, 100)
    X, Y = np.meshgrid(x, y)
    Z = function.f(X, Y)

    ax.contourf(X, Y, Z, cmap='viridis')
    ax.scatter(xlow, ylow, marker='x', color='blue', label='low risk samples(predicted)')
    ax.scatter(xhigh, yhigh, marker='x', color='red', label='high risk samples(predicted)')

    if mode == 'MHZ':
        start_point = (L-1, L-1)
    else:
        start_point = (l+1, l+1)

    a, b = a / np.sqrt(np.power(a, 2) + np.power(b, 2)), b / np.sqrt(np.power(a, 2) + np.power(b, 2))

    Length = np.sqrt(2) * (L - l) / 2.7
    a *= Length
    b *= Length
    direction_vector = (a, b)
    end_point = (start_point[0] + direction_vector[0], start_point[1] + direction_vector[1])
    ax.annotate("", xy=end_point, xytext=start_point,
                 arrowprops=dict(arrowstyle="->", color="orange", lw=2))
    ax.plot([], [], color='orange', lw=2, label='hazards increase')
    ax.legend(loc='best')
    ax.set_xlabel(f'Δx: samples {gene_name} expression (log2 transformed)')
    ax.set_ylabel(f'Δy: neighbors {gene_name} expression (log2 transformed)')
    ax.set_title(f'Hazards Contour plot [{name} model]')

# Process risk prediction for each layer
flags = []
for n in [1, 2, 4]:
    pt = f'{args.prediction_save_path}/{n}layers_model_prediction_risk.csv'
    risks = pd.read_csv(pt, header=0)['risk'].values
    md = np.median(risks)
    flag = np.zeros_like(risks)
    flag[risks > md] = 1  # Mark high-risk samples
    flag[risks < md] = 0  # Mark low-risk samples
    flag = flag.squeeze()
    flags.append(flag)

# Plot and save the hazard contour maps for each selected gene
for gene_idx, gene in enumerate(selected_genes):
    where = gene_names.index(gene)
    x = X[:, where]
    L = np.max(x)
    l = np.min(x)
    y = Y[:, where]

    # Create a figure with 3 subplots for 1-layer, 2-layer, and 4-layer models
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    model_name = ['1 layer Cox-Sage', '2 layers Cox-Sage', '4 layers Cox-Sage']
    for idx, alpha in enumerate(A):
        beta = B[idx]
        a, b = alpha[where], beta[where]
        xlow, ylow = x[flags[idx] == 0], y[flags[idx] == 0]
        xhigh, yhigh = x[flags[idx] == 1], y[flags[idx] == 1]
        plot_contour_map(axes[idx], a, b, xlow, ylow, xhigh, yhigh, l, L, gene, model_name[idx])

    # Save the plot
    plt.tight_layout()
    if mode == 'MHZ':
        plt.savefig(f'{graph_root}/low_expression_high_hazards/{gene}.png')
    else:
        plt.savefig(f'{graph_root}/high_expression_high_hazards/{gene}.png')





