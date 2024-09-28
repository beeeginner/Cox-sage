import pandas as pd
import torch
from torch.nn import init
from torch_geometric.nn import SAGEConv
import torch.nn as nn
import numpy as np
import pickle as pkl
import argparse
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, help="MHZ OR RMHZ")
parser.add_argument('--data_root', type=str, help="Root directory of the dataset",default="data")
parser.add_argument('--model_save_root', type=str, help="Root that you save trained models",default="save")
parser.add_argument('--dataset_name',type=str,help="The dataset you want to conduct prognstic gene discovery.",default="LIHC")
parser.add_argument('--best_folds', nargs='+', type=int, default=[4, 1, 2], help="Enter three numbers that indicate the fold at which each model achieved optimal performance; you can find this information in the training output log file.Default: 4 1 2, this is the best folds for the LIHC dataset.")
parser.add_argument('--output_graph_root', type=str, help="The root that you want to save the graph.",default='graphs')

args = parser.parse_args()
mode = args.mode
data_root = args.data_root
dataset_name = args.dataset_name
model_save_root = args.model_save_root
best_fold = args.best_folds
graph_root = args.output_graph_root
if not os.path.exists(graph_root):
    os.mkdir(graph_root)
    os.mkdir(os.path.join(graph_root,'low_expression_high_hazards'))
    os.mkdir(os.path.join(graph_root,'high_expression_high_hazards'))

assert mode in ('MHZ','RMHZ'),r'Mode must be set in MHZ or RMHZ'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df = pd.read_csv(f'{data_root}/{dataset_name}/gene_expression.csv',header=0)
with open(f'{data_root}/{dataset_name}/Adj_list.pkl','rb') as f:
    neibors = pkl.load(f)
del df['patient_id']
gene_names = df.columns.values.tolist()
X = df.values
X = np.log2(X+1)

class SageCox(nn.Module):
    def __init__(self, num_layers=4, eps=1e-5):
        super(SageCox, self).__init__()
        in_channels = X.shape[1]
        out_channels = 1
        self.num_layers = num_layers
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
            # 如果是两层
            self.GNN1 = SAGEConv(in_channels=in_channels, out_channels=hidden_channels)
            self.GNN2 = SAGEConv(in_channels=hidden_channels, out_channels=out_channels)
        elif num_layers == 1:
            # 如果只有一层
            self.GNN1 = SAGEConv(in_channels=in_channels, out_channels=out_channels)
        else:
            raise Exception
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
        if self._flag == 1:
            output = output.view(-1)
        return output

Y = np.zeros_like(X)
for v in range(X.shape[0]):
    neibor = neibors[v]
    Y[v] = np.mean(X[neibor],axis=0)

A,B = [],[]
layers = [1,2,4]
layers2fold = dict(zip(layers,best_fold))
for l in [1,2,4]:
    model = SageCox(num_layers=l).to(device)
    checkpoint = torch.load(f'{model_save_root}/{dataset_name}/{l}/{layers2fold[l]}fold_model_checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    # Get model parameters
    model_params = model.state_dict()
    if l==1:
        with torch.no_grad():
            alpha, beta = model_params['GNN1.lin_l.weight'].cpu().numpy(), model_params['GNN1.lin_l.weight'].cpu().numpy()
        A.append(alpha.squeeze())
        B.append(beta.squeeze())
    elif l==2:
        with torch.no_grad():
            W1, W2 = model_params['GNN1.lin_l.weight'], model_params['GNN1.lin_r.weight']
            alpha, beta = model_params['GNN2.lin_l.weight'], model_params[
                'GNN2.lin_l.weight']
            a = W1.T @ alpha.T
            b = W2.T @ beta.T
            del W1,W2,alpha,beta
            A.append(a.cpu().numpy().squeeze())
            B.append(b.cpu().numpy().squeeze())
    elif l==4:
        with torch.no_grad():
            W1, W2 = model_params['GNN1.lin_l.weight'], model_params['GNN1.lin_r.weight']
            W3, W4 = model_params['GNN2.lin_l.weight'], model_params['GNN2.lin_r.weight']
            W5, W6 = model_params['GNN3.lin_l.weight'], model_params['GNN3.lin_r.weight']
            alpha, beta = model_params['GNN4.lin_l.weight'], model_params[
                'GNN4.lin_l.weight']
            a = W1.T @ W3.T @ W5.T @ alpha.T
            del W1,W3,W5,alpha
            b = W2.T @ W4.T @ W6.T @ beta.T
            del W2,W4,W6,beta
            A.append(a.cpu().numpy().squeeze())
            B.append(b.cpu().numpy().squeeze())

all_selected_genes = []
for idx,l in enumerate([1,2,4]):
    a,b = A[idx],B[idx]
    delta_thetas = []
    for gene in range(X.shape[1]):
        expression_vector = X[:,gene].copy()
        delta_thetas.append(-np.mean(a[gene]*expression_vector+b[gene]*expression_vector))
    delta_thetas = np.array(delta_thetas)
    if mode=='MHZ':
        mhzs = np.exp(delta_thetas).tolist()
        df = pd.DataFrame({'gene_name':gene_names,'score':mhzs})
    else:
        rmhzs = np.exp(-delta_thetas).tolist()
        df = pd.DataFrame({'gene_name':gene_names,'score':rmhzs})

    scs = df['score'].values
    q3 = np.quantile(scs, q=0.75)
    q1 = np.quantile(scs, q=0.25)
    IQR = q3 - q1
    mid = np.quantile(scs, q=0.5)
    lb = mid + 1.5 * IQR
    df = df[df['score'] > lb].reset_index(drop=True)
    select = set(df['gene_name'].values.tolist())
    all_selected_genes.append(select)

print(f'1layer {len(all_selected_genes[0])} 2layers{len(all_selected_genes[1])} 4layers{len(all_selected_genes[2])}')
selected_genes = all_selected_genes[0] & all_selected_genes[1] & all_selected_genes[2]
selected_genes = list(selected_genes)

print(f'{mode} selecting {len(selected_genes)} genes in totall')
if mode=='MHZ':
    with open('gene_discovery.txt','a') as f:
        f.write(f'genes that low expression leading to high hazards: {selected_genes}\n')
else:
    with open('gene_discovery.txt','a') as f:
        f.write(f'genes that high expression leading to high hazards: {selected_genes}\n')

class binary_function:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def f(self, x, y):
        return np.exp(self.a * x + self.b * y)

def plot_contour_map(ax, a, b, xlow,ylow,xhigh,yhigh,l,L,gene_name,name):
    function = binary_function(a, b)
    x = np.linspace(l, L, 100)
    y = np.linspace(l, L, 100)
    X, Y = np.meshgrid(x, y)
    Z = function.f(X, Y)

    ax.contourf(X, Y, Z, cmap='viridis')
    ax.scatter(xlow,ylow,marker='x',color='blue',label='low risk samples(predicted)')
    ax.scatter(xhigh,yhigh,marker='x',color='red',label='high risk samples(predicted)')

    if mode=='MHZ':
        start_point = (L-1,L-1)
    else:
        start_point = (l+1,l+1)

    a,b = a/np.sqrt(np.power(a,2) +np.power(b,2)),b/np.sqrt(np.power(a,2) +np.power(b,2) )

    Length = np.sqrt(2)*(L-l)/2.7
    a*=Length
    b*=Length
    direction_vector = (a,b)
    end_point = (start_point[0] + direction_vector[0], start_point[1] + direction_vector[1])
    ax.annotate("", xy=end_point, xytext=start_point,
                 arrowprops=dict(arrowstyle="->", color="orange", lw=2))
    ax.plot([], [], color='orange', lw=2, label='hazards increases')
    ax.legend(loc='best')
    ax.set_xlabel(f'Δx: samples {gene_name} expression(log2 transformed)')
    ax.set_ylabel(f'Δy: neighbors {gene_name} expression(log2 transformed)')
    ax.set_title(f'Hazards Contour plot [{name} model]')

flags = []
for n in [1,2,4]:
    pt = f'{n}_risk.csv'
    risks = pd.read_csv(pt,header=0)['risk'].values
    md = np.median(risks)
    flag = np.zeros_like(risks)
    flag[risks>md]=1
    flag[risks<md]=0
    flag = flag.squeeze()
    flags.append(flag)

for gene_idx, gene in enumerate(selected_genes):
    where = gene_names.index(gene)
    x = X[:, where]
    L = np.max(x)
    l = np.min(x)
    y = Y[:, where]
    # Create a single figure and axis object
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    model_name = ['1 layer Cox-Sage','2 layers Cox-Sage','4 layers Cox-Sage']
    for idx, alpha in enumerate(A):
        beta = B[idx]
        a, b = alpha[where], beta[where]
        xlow,ylow = x[flags[idx]==0],y[flags[idx]==0]
        xhigh, yhigh = x[flags[idx] == 1], y[flags[idx] == 1]
        plot_contour_map(axes[idx], a,b,xlow,ylow,xhigh,yhigh,l,L,gene,model_name[idx])
    # Save the figure
    plt.tight_layout()
    if mode=='MHZ':
        plt.savefig(f'{graph_root}/low_expression_high_hazards/{gene}.png')
    else:
        plt.savefig(f'{graph_root}/high_expression_high_hazards/{gene}.png')



