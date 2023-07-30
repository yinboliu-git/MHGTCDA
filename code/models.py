import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, GCN2Conv, SAGEConv, GATConv, HGTConv, Linear,HANConv
from torch_geometric.utils import to_undirected
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from globel_args import device
# HGTConv = HANConv
def get_Hdata(x1, x2, e, ew=None):
    x1=x1.to(device)
    x2=x2.to(device)
    e=e.to(device)
    meta_dict = {
        'n1': {'num_nodes': x1.shape[0], 'num_features': x1.shape[1]},
        'n2': {'num_nodes': x2.shape[0], 'num_features': x2.shape[1]},
        ('n1', 'e1', 'n2'): {'edge_index': e, 'edge_weight': ew},
        ('n2', 'e1', 'n1'): {'edge_index': torch.flip(e, (0, )), 'edge_weight': ew},
    }

    data = HeteroData(meta_dict)
    data['n1'].x = x1
    data['n2'].x = x2
    data[('n1', 'e1', 'n2')].edge_index = e
    data[('n2', 'e1', 'n1')].edge_index = torch.flip(e, (0, ))

    data['x_dict'] = {ntype: data[ntype].x for ntype in data.node_types}
    edge_index_dict = {}
    for etype in data.edge_types:
        edge_index_dict[etype] = data[etype].edge_index
    data['edge_dict'] = edge_index_dict

    return data.to(device)

class HGT_old(torch.nn.Module):
    def __init__(self, hidden_channels, num_heads, num_layers, data):

        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels) # -1表示根据输入的数据的维度自动调整

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            #  in_channels: Union[int, Dict[str, int]],
            conv = HGTConv(-1, hidden_channels, data.metadata(),
                           num_heads, group='sum')
            self.convs.append(conv)

        self.fc = Linear(hidden_channels*2, 2)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x_dict_, edge_index_dict, edge_index):
        x_dict = x_dict_.copy()
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()
        all_list = []
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            all_list.append(x_dict.copy())

        for i,_ in x_dict_.items():
            x_dict[i] =torch.cat(tuple(x[i] for x in all_list), dim=1)

        m_index = edge_index[0]
        d_index = edge_index[1]
        #
        Em = self.dropout(x_dict['n1'])
        Ed = self.dropout(x_dict['n2'])
        y = Em@Ed.t()
        # y = torch.cat((Em, Ed), dim=1)
        # y = self.fc(y)
        y = y[m_index,d_index].unsqueeze(-1)
        return y

class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, num_heads, num_layers, data):

        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels) # -1表示根据输入的数据的维度自动调整

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            #  in_channels: Union[int, Dict[str, int]],
            conv = HGTConv(-1, hidden_channels, data.metadata(),
                           num_heads, group='sum')
            self.convs.append(conv)

        self.fc = Linear(hidden_channels*2, 2)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x_dict_, edge_index_dict, edge_index):
        x_dict = x_dict_.copy()
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()
        all_list = []
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            all_list.append(x_dict.copy())

        for i,_ in x_dict_.items():
            x_dict[i] =torch.cat(tuple(x[i] for x in all_list), dim=1)

        m_index = edge_index[0]
        d_index = edge_index[1]
        #
        Em = self.dropout(x_dict['n1'])
        Ed = self.dropout(x_dict['n2'])
        y = Em@Ed.t()
        # y = torch.cat((Em, Ed), dim=1)
        # y = self.fc(y)
        y = y[m_index,d_index].unsqueeze(-1)
        return y

