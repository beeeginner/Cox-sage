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
            "label_csv" is not the same file as "label.csv". Here, "label_csv" refers to the clinical data file.
        '''

        assert train_size + test_size == 1, 'The sum of the proportions of the training set and test set must be equal to 1.'
        self.train_size = train_size
        self.test_size = test_size
        self.path_node = node_csv
        with open(adj_list, 'rb') as f:
            self.adj_list = pkl.load(f)
        self.label_csv = label_csv
        self.device = device

    def _get_edge_index(self):
        '''
            Get edge information using an adjacency list.
        '''
        edge_index = []
        for key, items in self.adj_list.items():
            for edge2 in items:
                edge_index.append([key, edge2])
        edge_index = np.array(edge_index, dtype=np.int64).T
        edge_index = torch.from_numpy(edge_index)
        return edge_index

    def process(self):
        node_file = self.path_node
        df_node = pd.read_csv(node_file, header=0)
        self.name_list = df_node['patient_id'].values.tolist()
        del df_node['patient_id']
        X = df_node.values.astype('float32')
        X = np.log2(X + 1)
        vital_status = pd.read_csv(self.label_csv, header=0)['vital_status'].values.astype('float32')
        survival_time = pd.read_csv(self.label_csv, header=0)['real_survival_time'].values.astype('float32')
        edge_index = self._get_edge_index()
        data = Data(x=torch.from_numpy(X),
                    edge_index=edge_index)
        data.vital_status = torch.from_numpy(vital_status)
        data.survival_time = torch.from_numpy(survival_time)
        data = data.to(self.device)
        return data
def loss_function(thetas, survival_time, events):
    loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    Events_index = torch.nonzero(events == 1).squeeze()
    for i in Events_index:
        loss += thetas[i]
        risk_sum = torch.sum(torch.exp(thetas[survival_time >= survival_time[i]]))
        loss -= torch.log(risk_sum)

    return -loss

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

def stratified_random_partition(arr, partitions=10, stratify_labels=None, random_state=None):
    if not isinstance(arr,np.ndarray):
        arr = np.array(arr)
    if stratify_labels is None:
        raise ValueError(r"stratify_labels must be provided for stratified sampling.")
    skf = StratifiedKFold(n_splits=partitions, shuffle=True, random_state=random_state)
    # 生成分层交叉采样
    fold_indices = list(skf.split(arr, stratify_labels))
    groups = [[] for _ in range(partitions)]
    for i, (_, test_indices) in enumerate(fold_indices):
        groups[i] = arr[test_indices].tolist()
    return groups

def train(gene,adj_list,clinical,num_layers,save_path,log,seed=0,lr=1e-5,wd=1e-3):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MyOwnDataset(node_csv=gene, \
                           adj_list=adj_list, label_csv=clinical, \
                           device=device)
    data = dataset.process()
    class SageCox(nn.Module):
        def __init__(self, num_layers = 4, eps=1e-5):
            super(SageCox, self).__init__()
            in_channels = data.x.shape[1]
            out_channels = 1
            self.num_layers = num_layers
            if num_layers==4:
                n1 = int(in_channels*(2/3))
                n2 = int(n1*(2/3))
                n3 = int(n2/2)
                self.GNN1 = SAGEConv(in_channels=in_channels, out_channels=n1)
                self.GNN2 = SAGEConv(in_channels=n1, out_channels=n2)
                self.GNN3 = SAGEConv(in_channels=n2, out_channels=n3)
                self.GNN4 = SAGEConv(in_channels=n3, out_channels=out_channels)
            elif num_layers == 2:
                hidden_channels = int(in_channels/2)
                # 如果是两层
                self.GNN1 = SAGEConv(in_channels=in_channels, out_channels=hidden_channels)
                self.GNN2 = SAGEConv(in_channels=hidden_channels, out_channels=out_channels)
            elif num_layers == 1:
                # 如果只有一层
                self.GNN1 = SAGEConv(in_channels=in_channels, out_channels=out_channels)
            else:
                raise Exception


        def forward(self, data):
            x, edge_index  = data.x, data.edge_index
            if self.num_layers==4:
                output = self.GNN1(x=x, edge_index=edge_index)
                output = self.GNN2(x=output, edge_index=edge_index)
                output = self.GNN3(x=output, edge_index=edge_index)
                output = self.GNN4(x=output, edge_index=edge_index)
            elif self.num_layers==2:
                output = self.GNN1(x=x, edge_index=edge_index)
                output = self.GNN2(x=output, edge_index=edge_index)
            else:
                output = self.GNN1(x=x, edge_index=edge_index)
            output = output.view(-1)

            return output
    k = 5
    epochs = 20
    early_stopping_patience = 3

    N = data.x.shape[0]
    all_samples = [i for i in range(N)]
    train_samples, val_samples = train_test_split(all_samples, train_size=0.8, random_state=seed,
                                                  stratify=data.vital_status.cpu())

    val_mask = torch.BoolTensor([False for i in range(N)])
    val_samples = torch.LongTensor(val_samples)
    val_mask[val_samples] = True
    train_mask = torch.BoolTensor([False for i in range(N)])
    train_samples = torch.LongTensor(train_samples)
    train_mask[train_samples] = True
    test_samples = stratified_random_partition(train_samples, partitions=k,
                                               stratify_labels=data.vital_status.cpu()[train_mask], random_state=seed)
    train_samples = set(train_samples)

    c_indices = []
    for idx, test in enumerate(test_samples):
        test = set(test)
        train = train_samples - test
        train, test = torch.tensor(list(train), dtype=torch.long), torch.tensor(list(test), dtype=torch.long)
        train_mask = torch.BoolTensor([False for i in range(N)])
        test_mask = torch.BoolTensor([False for i in range(N)])
        train_mask[train] = True
        test_mask[test] = True
        model = SageCox(num_layers=num_layers).to(device)
        optimizer = Adam(model.parameters(), lr=lr,weight_decay=wd)
        model.train()
        best_c_index = -torch.inf
        counter = 0
        print(f'{k}fold:{idx + 1}/{k}')
        for epoch in range(epochs):
            if counter > early_stopping_patience:
                print(f'Early stopping triggered!Finially epoch:{epoch}')
                break
            model.train()
            optimizer.zero_grad()
            out = model(data).squeeze()
            train_loss = loss_function(out[train_mask],data.survival_time[train_mask],data.vital_status[train_mask])
            train_loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.eval()
                out = model(data)
                test_loss = loss_function(out[test_mask],data.survival_time[test_mask],data.vital_status[test_mask])
                train_index, test_index = c_index(out[train_mask], data.survival_time[train_mask],
                                                  data.vital_status[train_mask]), c_index(out[test_mask],
                                                                                          data.survival_time[
                                                                                              test_mask],
                                                                                          data.vital_status[
                                                                                              test_mask])
                val_index = c_index(out[val_mask], data.survival_time[val_mask], data.vital_status[val_mask])
                print(f'{k}fold:{idx + 1}/{k} epoch: {epoch}, train loss: {train_loss.item()}, test loss: {test_loss.item()},train c-index: {train_index},test c-index: {test_index},val c-index:{val_index}')
                if val_index <= best_c_index:
                    counter += 1
                else:
                    counter = 0
                    best_c_index = val_index
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, os.path.join(save_path,f'{idx+1}fold_model_checkpoint.pth'))
                    with open(os.path.join(save_path,f'{idx+1}fold_train_mask.pkl'), 'wb') as f:
                        pkl.dump(train_mask.cpu(), f)
                    with open(os.path.join(save_path,f'{idx+1}fold_test_mask.pkl'), 'wb') as f:
                        pkl.dump(test_mask.cpu(), f)
        c_indices.append(best_c_index)

    c_indices = np.array(c_indices)
    kfoldidx = np.argmax(c_indices) + 1
    best_model = SageCox(num_layers=num_layers).to(device)
    checkpoint = torch.load(os.path.join(save_path,f'{kfoldidx}fold_model_checkpoint.pth'))
    best_model.load_state_dict(checkpoint['model_state_dict'])
    best_model.eval()
    with torch.no_grad():
        out = best_model(data)
        val_index1 = c_index(out[val_mask], data.survival_time[val_mask], data.vital_status[val_mask])
    print(f'C-index computed in val dataset is {val_index1}')

    with open(log,'a') as f:
        f.write(f'model: {save_path} layers:{num_layers} C-index: {val_index1}\n')

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








