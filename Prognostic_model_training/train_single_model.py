import pandas as pd
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import torch
from torch_geometric.nn import SAGEConv
import torch.nn as nn
import pickle as pkl
from torch.optim import Adam
import os
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class MyOwnDataset:

    def __init__(self, node_csv, adj_list, label_csv, device='cpu', train_size=0.7, test_size=0.3):
        '''
            label_csv is not the label.csv file, here label_csv refers to the clinical data file
        '''

        assert train_size + test_size == 1, 'The sum of the train and test proportions must be 1'
        self.train_size = train_size
        self.test_size = test_size
        self.path_node = node_csv
        with open(adj_list, 'rb') as f:
            self.adj_list = pkl.load(f)
        self.label_csv = label_csv
        self.device = device  # The device where the data is stored

    def _get_edge_index(self):
        '''
            Get edge information using the adjacency list
        '''
        edge_index = []
        for key, items in self.adj_list.items():
            for edge2 in items:
                edge_index.append([key, edge2])
        edge_index = np.array(edge_index, dtype=np.int64).T
        edge_index = torch.from_numpy(edge_index)
        return edge_index

    def process(self):
        # Path to the graph nodes
        node_file = self.path_node
        # Remove the node name column
        df_node = pd.read_csv(node_file, header=0)
        self.name_list = df_node['patient_id'].values.tolist()
        del df_node['patient_id']
        # Build the graph
        X = df_node.values.astype('float32')  # Features
        X = np.log2(X + 1)  # Log transformation]
        vital_status = pd.read_csv(self.label_csv, header=0)['vital_status'].values.astype('float32')  # Risk values
        survival_time = pd.read_csv(self.label_csv, header=0)['real_survival_time'].values.astype('float32')  # Risk values
        edge_index = self._get_edge_index()
        data = Data(x=torch.from_numpy(X),
                    edge_index=edge_index)
        data.vital_status = torch.from_numpy(vital_status)
        data.survival_time = torch.from_numpy(survival_time)
        data = data.to(self.device)
        return data

def loss_function(thetas, survival_time, events):
    loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    Events_index = torch.nonzero(events == 1).squeeze()  # Indices of samples where events occurred

    for i in Events_index:
        loss += thetas[i]  # First term of the log-likelihood
        # Accumulated risk term, iterate over all samples with survival time greater than or equal to t_i
        risk_sum = torch.sum(torch.exp(thetas[survival_time >= survival_time[i]]))
        loss -= torch.log(risk_sum)  # Second term of the log-likelihood: log of accumulated risk

    return -loss  # Minimize negative log-likelihood

# Calculate C-index
def c_index(risk_pred, survival_time, events):

    risk_pred = risk_pred.detach().cpu().numpy()
    survival_time = survival_time.detach().cpu().numpy()
    events = events.detach().cpu().numpy()

    n = len(risk_pred)
    numerator = 0
    denumerator = 0

    def I(statement):
        return 1 if statement else 0

    for i in range(n):
        for j in range(i + 1, n):
            numerator += events[i] * I(survival_time[i] < survival_time[j]) * I(risk_pred[i] > risk_pred[j])
            denumerator += events[i] * I(survival_time[i] < survival_time[j])
    c_index = numerator / denumerator
    return c_index

# Stratified cross-validation sampling
def stratified_random_partition(arr, partitions=10, stratify_labels=None, random_state=None):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if stratify_labels is None:
        raise ValueError("stratify_labels must be provided for stratified sampling")
    skf = StratifiedKFold(n_splits=partitions, shuffle=True, random_state=random_state)
    # Generate stratified cross-validation samples
    fold_indices = list(skf.split(arr, stratify_labels))
    groups = [[] for _ in range(partitions)]
    for i, (_, test_indices) in enumerate(fold_indices):
        groups[i] = arr[test_indices].tolist()
    return groups

def train(gene, adj_list, clinical, num_layers, save_path, log, seed=0, lr=1e-5, wd=1e-3):

    torch.manual_seed(seed)
    # If you are using GPU, you also need to set the following code to ensure reproducibility
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set NumPy random seed
    np.random.seed(seed)
    # Decide which device to compute on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MyOwnDataset(node_csv=gene, \
                           adj_list=adj_list, label_csv=clinical, \
                           device=device)
    data = dataset.process()

    class SageCox(nn.Module):
        def __init__(self, num_layers=4, eps=1e-5):
            super(SageCox, self).__init__()
            in_channels = data.x.shape[1]
            out_channels = 1
            self.num_layers = num_layers
            if num_layers == 4:
                n1 = int(in_channels * (2/3))
                n2 = int(n1 * (2/3))
                n3 = int(n2 / 2)
                self.GNN1 = SAGEConv(in_channels=in_channels, out_channels=n1)
                self.GNN2 = SAGEConv(in_channels=n1, out_channels=n2)
                self.GNN3 = SAGEConv(in_channels=n2, out_channels=n3)
                self.GNN4 = SAGEConv(in_channels=n3, out_channels=out_channels)
            elif num_layers == 2:
                hidden_channels = int(in_channels / 2)
                # If there are two layers
                self.GNN1 = SAGEConv(in_channels=in_channels, out_channels=hidden_channels)
                self.GNN2 = SAGEConv(in_channels=hidden_channels, out_channels=out_channels)
            elif num_layers == 1:
                # If there is only one layer
                self.GNN1 = SAGEConv(in_channels=in_channels, out_channels=out_channels)
            else:
                raise Exception
            self.eps = eps

        def forward(self, data):
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
            return output

    k = 5
    epochs = 20  # Number of training epochs per iteration
    early_stopping_patience = 3  # Early stopping tolerance

    # Split validation set
    N = data.x.shape[0]  # Total number of samples
    all_samples = [i for i in range(N)]  # All samples
    train_samples, val_samples = train_test_split(all_samples, train_size=0.8, random_state=seed,
                                                  stratify=data.vital_status.cpu())  # Split validation set
    # Compute validation set mask
    val_mask = torch.BoolTensor([False for i in range(N)])
    val_samples = torch.LongTensor(val_samples)
    val_mask[val_samples] = True
    # Train and test set masks
    train_mask = torch.BoolTensor([False for i in range(N)])
    train_samples = torch.LongTensor(train_samples)
    train_mask[train_samples] = True
    # Get k-fold cross-validation test sets
    test_samples = stratified_random_partition(train_samples, partitions=k,
                                               stratify_labels=data.vital_status.cpu().numpy(),
                                               random_state=seed)
    test_samples = [torch.LongTensor(test) for test in test_samples]  # Test set sample IDs
    optimizer = torch.optim.Adam(params=SageCox.parameters(), lr=lr, weight_decay=wd)

    for fold in range(k):  # Train for each fold
        # Initialize the model
        SageCox = SageCox(num_layers=num_layers)
        best_val_loss = float('inf')  # Record the best validation set loss
        patience_counter = 0  # Early stopping patience counter
        # Get fold's test set mask
        test_mask = torch.BoolTensor([False for i in range(N)])
        test_mask[test_samples[fold]] = True
        # Create training mask by excluding the current test set
        fold_train_mask = ~test_mask

        for epoch in range(epochs):
            SageCox.train()
            optimizer.zero_grad()  # Clear gradients

            # Get the model output (predictions)
            risk_pred = SageCox(data)
            loss = loss_function(risk_pred[fold_train_mask], data.survival_time[fold_train_mask],
                                 data.vital_status[fold_train_mask])

            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters

            # Evaluate on the validation set
            SageCox.eval()
            with torch.no_grad():  # Do not compute gradients
                val_pred = SageCox(data)
                val_loss = loss_function(val_pred[val_mask], data.survival_time[val_mask],
                                         data.vital_status[val_mask])

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save the model
                torch.save(SageCox.state_dict(), save_path + f'fold_{fold}.pt')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f'Early stopping at epoch {epoch}')
                    break

        # Evaluate on the test set
        SageCox.eval()
        with torch.no_grad():  # Do not compute gradients
            test_pred = SageCox(data)
            test_loss = loss_function(test_pred[test_mask], data.survival_time[test_mask],
                                      data.vital_status[test_mask])

            print(f'Test loss for fold {fold}: {test_loss.item()}')

            # Compute the C-index
            c_index_score = c_index(test_pred[test_mask], data.survival_time[test_mask],
                                    data.vital_status[test_mask])
            print(f'C-index for fold {fold}: {c_index_score}')

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',type=str,required=True,help="The root path of datasets")
    parser.add_argument('--model_save_root',type=str,required=True,help="The root path of datasets")
    parser.add_argument('--disease',type=str,required=True,help="The dataset that you want to train")
    parser.add_argument('--num_layers',type=str,required=True,help="The layers that you want for the Cox-sage model")
    parser.add_argument('--log_path',type=str,required=True,help="Log file")
    parser.add_argument('--seed',type=str,required=False,help="random seed default: 0")
    parser.add_argument('--lr',type=str,required=False,help="learning rate default: 1e-5")
    parser.add_argument('--wd',type=str,required=False,help="weight decay default: 1e-3")

    args = parser.parse_args()
    data_root = args.data_root
    disease = args.disease
    num_layers = eval(args.num_layers)
    model_save_root = args.model_save_root
    log = args.log_path
    disease_root = os.path.join(model_save_root,disease)
    if not os.path.exists(disease_root):
        os.mkdir(disease_root)

    current_path = os.path.join(data_root,disease)
    gene_path = os.path.join(current_path,'gene_expression.csv')
    adj_list_path = os.path.join(current_path,'Adj_list.pkl')
    clinical_path = os.path.join(current_path,'clinical.csv')
    model_save = os.path.join(disease_root,str(num_layers))
    os.mkdir(model_save)

    seed = args.seed
    if not seed:
        seed = 0
    else:
        seed = eval(seed)
    lr,wd = args.lr,args.wd
    if not lr:
        lr = 1e-5
    else:
        lr = eval(lr)
    if not wd:
        wd = 1e-3
    else:
        wd = eval(wd)

    train(gene_path,adj_list_path,clinical_path,num_layers,model_save,log,seed,lr,wd)
