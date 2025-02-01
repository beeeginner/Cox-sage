# Import required libraries
from torch_geometric.data import Data  # PyTorch Geometric data handling
import numpy as np  # Numerical computing
import torch  # PyTorch deep learning framework
from torch_geometric.nn import SAGEConv  # Graph SAGE convolutional layer
import torch.nn as nn  # Neural network modules
import pickle as pkl  # Object serialization
import os  # File system operations
import argparse  # Command-line argument parsing
import pandas as pd  # Data manipulation
import matplotlib.pyplot as plt  # Data visualization
from lifelines import KaplanMeierFitter  # Survival analysis

# Set device to CUDA if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Dataset Class for handling graph-structured medical data
class MyOwnDataset:
    def __init__(self, node_csv, adj_list, label_csv, device='cpu', train_size=0.7, test_size=0.3):
        '''
        Initialize dataset parameters and paths.
        
        Args:
            node_csv (str): Path to node features CSV (gene expression data)
            adj_list (str): Path to adjacency list pickle file (graph structure)
            label_csv (str): Path to clinical data CSV (survival labels)
            device (str): Computation device ('cpu' or 'cuda')
            train_size (float): Proportion of training data
            test_size (float): Proportion of test data
        '''
        assert train_size + test_size == 1, 'Train/test split must sum to 1'
        self.train_size = train_size
        self.test_size = test_size
        self.path_node = node_csv
        with open(adj_list, 'rb') as f:
            self.adj_list = pkl.load(f)  # Load adjacency list from pickle
        self.label_csv = label_csv
        self.device = device

    def _get_edge_index(self):
        '''
        Convert adjacency list to edge index tensor format required by PyTorch Geometric.
        
        Returns:
            torch.Tensor: Edge index tensor of shape [2, num_edges]
        '''
        edge_index = []
        for key, items in self.adj_list.items():
            for edge2 in items:
                edge_index.append([key, edge2])
        edge_index = np.array(edge_index, dtype=np.int64).T  # Transpose to [2, num_edges]
        return torch.from_numpy(edge_index)

    def process(self):
        '''
        Process and combine all data components into a PyTorch Geometric Data object.
        
        Returns:
            Data: Processed graph data containing:
                - x: Node features (gene expression)
                - edge_index: Graph connectivity
                - vital_status: Survival status labels
                - survival_time: Time-to-event data
        '''
        # Load and process node features
        df_node = pd.read_csv(self.path_node).sort_values('patient_id')
        self.name_list = df_node['patient_id'].tolist()
        X = df_node.drop('patient_id', axis=1).values.astype('float32')
        X = np.log2(X + 1)  # Log-transform gene expression values

        # Load and process clinical labels
        df_clinical = pd.read_csv(self.label_csv).sort_values('patient_id')
        vital_status = df_clinical['vital_status'].values.astype('float32')
        survival_time = df_clinical['real_survival_time'].values.astype('float32')

        # Create PyTorch Geometric Data object
        data = Data(
            x=torch.from_numpy(X),
            edge_index=self._get_edge_index(),
            vital_status=torch.from_numpy(vital_status),
            survival_time=torch.from_numpy(survival_time)
        )
        return data.to(self.device)  # Transfer data to specified device

# Graph Neural Network for Survival Analysis
class SageCox(nn.Module):
    def __init__(self, num_layers=4, eps=1e-5):
        '''
        Initialize CoxSage model with specified number of graph layers.
        
        Args:
            num_layers (int): Number of SAGEConv layers (1, 2, or 4)
            eps (float): Small value for numerical stability
        '''
        super().__init__()
        in_channels = data.x.shape[1]  # Input feature dimension
        out_channels = 1  # Single output node for risk score
        
        # Configure layer dimensions based on number of layers
        if num_layers == 4:
            n1 = int(in_channels*(2/3))
            n2 = int(n1*(2/3))
            n3 = int(n2/2)
            self.layers = nn.ModuleList([
                SAGEConv(in_channels, n1),
                SAGEConv(n1, n2),
                SAGEConv(n2, n3),
                SAGEConv(n3, out_channels)
            ])
        elif num_layers == 2:
            hidden = int(in_channels/2)
            self.layers = nn.ModuleList([
                SAGEConv(in_channels, hidden),
                SAGEConv(hidden, out_channels)
            ])
        elif num_layers == 1:
            self.layers = nn.ModuleList([SAGEConv(in_channels, out_channels)])
        else:
            raise ValueError("Supported layers: 1, 2, or 4")
            
        self.num_layers = num_layers
        self.eps = eps

    def forward(self, data):
        '''
        Forward pass through the network.
        
        Args:
            data (Data): Input graph data
        Returns:
            torch.Tensor: Predicted risk scores
        '''
        x, edge_index = data.x, data.edge_index
        for layer in self.layers:
            x = layer(x, edge_index)
        return x

# Model Inference Function
def inference(gene, adj_list, clinical, num_layers, model_path):
    '''
    Run inference using trained model.
    
    Args:
        gene (str): Path to gene expression CSV
        adj_list (str): Path to adjacency list pickle
        clinical (str): Path to clinical data CSV
        num_layers (int): Number of layers in trained model
        model_path (str): Path to saved model weights
    
    Returns:
        numpy.ndarray: Predicted risk scores
    '''
    # Initialize dataset and model
    dataset = MyOwnDataset(gene, adj_list, clinical, device=device)
    data = dataset.process()
    
    # Load trained model
    model = SageCox(num_layers).to(device)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()  # Set to evaluation mode

    # Generate predictions
    with torch.no_grad():
        risk_scores = model(data).cpu().numpy()
    
    return np.squeeze(risk_scores)  # Remove singleton dimensions

# Main Execution Block
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Survival Prediction using Graph Neural Networks')
    parser.add_argument('--disease_name', required=True, help='Name of disease cohort')
    parser.add_argument('--dataset_root', required=True, help='Root directory for datasets')
    parser.add_argument('--num_layers', required=True, help='Number of GNN layers (1, 2, or 4)')
    parser.add_argument('--model_root', required=True, help='Directory containing trained models')
    parser.add_argument('--prediction_output_dir', required=True, help='Output directory for results')
    args = parser.parse_args()

    # Configure paths
    data_dir = os.path.join(args.dataset_root, args.disease_name)
    model_dir = os.path.join(args.model_root, args.disease_name, str(args.num_layers))
    
    # File paths setup
    gene_path = os.path.join(data_dir, 'gene_expression.csv')
    adj_path = os.path.join(data_dir, 'Adj_list.pkl')
    clinical_path = os.path.join(data_dir, 'clinical.csv')
    model_path = os.path.join(model_dir, 'best_fold.pth')

    # Generate risk predictions
    risk_scores = inference(gene_path, adj_path, clinical_path, 
                           num_layers=int(args.num_layers), 
                           model_path=model_path)
    
    # Post-process risk scores
    risk_z = (risk_scores - np.mean(risk_scores)) / np.std(risk_scores)  # Z-score normalization
    risk_probs = 1 / (1 + np.exp(-risk_z))  # Sigmoid transformation to probabilities
    
    # Save predictions
    pd.DataFrame({'risk': risk_probs}).to_csv(
        f'{args.prediction_output_dir}/{args.num_layers}layers_prediction.csv',
        index=False
    )

    # Survival Analysis Visualization
    clinical_data = pd.read_csv(clinical_path)
    survival_time = clinical_data.sort_values('patient_id')['real_survival_time']
    vital_status = clinical_data.sort_values('patient_id')['vital_status']
    
    # Split into risk groups using median threshold
    median_risk = np.median(risk_probs)
    low_risk_mask = risk_probs <= median_risk
    high_risk_mask = risk_probs > median_risk

    # Kaplan-Meier Curve Plotting
    plt.figure(figsize=(10, 5))
    kmf = KaplanMeierFitter()
    
    # Plot low-risk group
    kmf.fit(survival_time[low_risk_mask], vital_status[low_risk_mask])
    kmf.plot(label='Low Risk', ci_show=False)
    
    # Plot high-risk group
    kmf.fit(survival_time[high_risk_mask], vital_status[high_risk_mask])
    kmf.plot(label='High Risk', ci_show=False)
    
    plt.title(f'Survival Analysis ({args.num_layers}-layer Model)')
    plt.xlabel('Survival Time (days)')
    plt.ylabel('Survival Probability')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{args.prediction_output_dir}/KM_curve_{args.num_layers}layers.png')
    
    # Log-Rank Test for Statistical Significance
    from lifelines.statistics import logrank_test
    result = logrank_test(
        survival_time[low_risk_mask], 
        survival_time[high_risk_mask],
        vital_status[low_risk_mask],
        vital_status[high_risk_mask]
    )
    print(result.print_summary())
