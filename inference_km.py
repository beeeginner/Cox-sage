from torch_geometric.data import Data
import numpy as np
import torch
from torch_geometric.nn import SAGEConv
import torch.nn as nn
import pickle as pkl
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

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
        df_node = df_node.sort_values(by='patient_id')
        df_node = df_node.reset_index(drop=True)
        df_clinical = pd.read_csv(self.label_csv, header=0)
        df_clinical = df_clinical.sort_values(by='patient_id')
        df_clinical = df_clinical.reset_index(drop=True)
        self.name_list = df_node['patient_id'].values.tolist()
        del df_node['patient_id']
        X = df_node.values.astype('float32')
        X = np.log2(X + 1)
        vital_status = df_clinical['vital_status'].values.astype('float32')
        survival_time = df_clinical['real_survival_time'].values.astype('float32')
        edge_index = self._get_edge_index()
        data = Data(x=torch.from_numpy(X),
                    edge_index=edge_index)
        data.vital_status = torch.from_numpy(vital_status)
        data.survival_time = torch.from_numpy(survival_time)
        data = data.to(self.device)
        return data
def inference(gene,adj_list,clinical,num_layers,model_path):

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

                self.GNN1 = SAGEConv(in_channels=in_channels, out_channels=hidden_channels)
                self.GNN2 = SAGEConv(in_channels=hidden_channels, out_channels=out_channels)
            elif num_layers == 1:
                self.GNN1 = SAGEConv(in_channels=in_channels, out_channels=out_channels)
            else:
                raise Exception
            self.eps = eps


        def forward(self, data):
            x, edge_index = data.x, data.edge_index
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
            # output = torch.exp(output)
            return output

    best_model = SageCox(num_layers=num_layers).to(device)
    checkpoint = torch.load(model_path)
    best_model.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        out = best_model(data)

    return out

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--disease_name', type=str, required=True, help="disease name")
    parser.add_argument('--dataset_root',type=str,required=True,help="dataset root")
    parser.add_argument('--num_layers', type=str, required=True, help="num layers of the model")
    parser.add_argument('--model_root', type=str, required=True, help="model path")
    parser.add_argument('--prediction_output_dir', type=str, required=True, help="directory to storage prediction results")

    args = parser.parse_args()
    num_layers = eval(args.num_layers)
    data_root = args.dataset_root
    disease_name = args.disease_name
    dataset_path = os.path.join(data_root,disease_name)
    gene = os.path.join(dataset_path,'gene_expression.csv')
    adj_list = os.path.join(dataset_path,'Adj_list.pkl')
    clinical = os.path.join(dataset_path,'clinical.csv')

    model_root = args.model_root
    model_root = os.path.join(model_root,disease_name)
    model_path = os.path.join(model_root,str(num_layers))+'/'+'best_fold.pth'

    pred_risk = np.squeeze(inference(gene,adj_list,clinical,num_layers,model_path).cpu().numpy())
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    mean = np.mean(pred_risk)
    std = np.std(pred_risk)
    pred_risk_z = (pred_risk - mean) / std

    pred_risk = sigmoid(pred_risk_z)

    pd.DataFrame({'risk':pred_risk.tolist()}).to_csv(f'{args.prediction_output_dir}/{num_layers}layers_model_prediction_risk.csv',index=False)
    survival_time = pd.read_csv(clinical, header=0).sort_values(by='patient_id')['real_survival_time']
    status = pd.read_csv(clinical, header=0).sort_values(by='patient_id')['vital_status']

    median_risk = np.median(pred_risk)

    group_1_mask = pred_risk <= median_risk
    group_2_mask = pred_risk > median_risk

    kmf = KaplanMeierFitter()

    kmf.fit(survival_time[group_1_mask.flatten()], event_observed=status[group_1_mask.flatten()])
    plt.figure(figsize=(10, 5))
    kmf.plot(label='Low Risk Group')
    plt.title('Kaplan-Meier Curve')
    plt.xlabel('Days')
    plt.ylabel('Survival Probability')
    plt.grid(True)

    # 计算第二组的 Kaplan-Meier 曲线
    kmf.fit(survival_time[group_2_mask.flatten()], event_observed=status[group_2_mask.flatten()])
    kmf.plot(label='High Risk Group')
    plt.xlabel('Days')
    plt.ylabel('Survival Probability')
    plt.grid(True)

    plt.legend()
    plt.savefig(f'{args.prediction_output_dir}/Cox-sage({num_layers}layers).png')

    from lifelines.statistics import logrank_test

    results = logrank_test(survival_time[group_1_mask.flatten()], survival_time[group_2_mask.flatten()],
                           event_observed_A=status[group_1_mask.flatten()],
                           event_observed_B=status[group_2_mask.flatten()])
    print(results.print_summary())
