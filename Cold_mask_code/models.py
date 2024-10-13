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
    # data['xe'] = {'n1': xe1, 'n2':xe2}


    return data.to(device)


class GCNII(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(GCNII, self).__init__()
        self.convs = torch.nn.ModuleList()
        conv = GCNConv(in_channels, out_channels)
        self.convs.append(conv)
        for _ in range(num_layers-1):
            conv = GCNConv(out_channels, out_channels)
            self.convs.append(conv)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)

        return x


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, num_heads, num_layers, data, params):

        super().__init__()
        self.params = params
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels) # -1表示根据输入的数据的维度自动调整

        if self.params.ablation == 'no-HGT':
            HNN = HANConv
        elif self.params.ablation != 'no-HNN':
            HNN = HGTConv

        self.convs = torch.nn.ModuleList()
        if self.params.ablation != 'no-HNN':
            for _ in range(num_layers):
                #  in_channels: Union[int, Dict[str, int]],
                conv = HNN(-1, hidden_channels, data.metadata(),
                               num_heads, group='sum')
                self.convs.append(conv)

        if self.params.ablation == 'no-HNN':
            for _ in range(num_layers):
                conv = GCNII(-1, hidden_channels, num_heads)
                self.convs.append(conv)

        self.fc = Linear(-1, 1)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x_dict_, edge_index_dict, edge_index):
        x_dict = x_dict_.copy()
        if self.params.ablation == 'no-FNN':
            for node_type, x in x_dict.items():
                x_dict[node_type] = x

        elif self.params.ablation != 'no-rand':
            for node_type, x in x_dict.items():
                x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        if self.params.ablation == 'no-HNN':
            e_all = []
            for e_type, e in edge_index_dict.items():
                e_all = torch.clone(e)
                break

            x_all = []
            x_len = []
            for node_type, _ in x_dict.items():
                x_len.append(x_dict[node_type].shape[0])

                x_all.append(x_dict[node_type])

            e_all[1,:] += x_len[0]
            e_all = to_undirected(e_all)

            x_all = torch.cat(tuple(x_all), dim=0)

            all_list = []
            for conv in self.convs:
                x_all = conv(x_all, e_all)
                x_dict = x_dict_.copy()
                ctrl_i = 0
                for node_type, x in x_dict.items():
                    if ctrl_i == 0:
                        x_dict[node_type] = x_all[:x_len[0],:]
                        ctrl_i += 1
                    else:
                        x_dict[node_type] = x_all[x_len[0]:,:]
                all_list.append(x_dict)

        else:
            all_list = []
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)
                all_list.append(x_dict.copy())

        m_index = edge_index[0]
        d_index = edge_index[1]
        if self.params.ablation == 'no-catmul':
            for i,_ in x_dict_.items():
                x_dict[i] =torch.cat(tuple(x[i] for x in all_list), dim=1)
            Em = self.dropout(torch.index_select(x_dict['n1'], 0, m_index))  # 沿着x1的第0维选出m_index
            Ed = self.dropout(torch.index_select(x_dict['n2'], 0, d_index))
            x = torch.cat((Em, Ed), dim=1)
            y = self.fc(x)

        elif self.params.ablation == 'en-mean':
            for i, _ in x_dict_.items():
                x_dict[i] =torch.cat(tuple(x[i].unsqueeze(0) for x in all_list), dim=0).mean(dim=0)

            m_index = edge_index[0]
            d_index = edge_index[1]
            #
            Em = self.dropout(x_dict['n1'])
            Ed = self.dropout(x_dict['n2'])
            y = Em@Ed.t()
            y = y[m_index, d_index].unsqueeze(-1)

        elif self.params.ablation == 'no-H4':

            m_index = edge_index[0]
            d_index = edge_index[1]
            #
            Em = self.dropout(x_dict['n1'])
            Ed = self.dropout(x_dict['n2'])
            y = Em@Ed.t()
            y = y[m_index, d_index].unsqueeze(-1)

        else:
            for i, _ in x_dict_.items():
                x_dict[i] = torch.cat(tuple(x[i] for x in all_list), dim=1)

            m_index = edge_index[0]
            d_index = edge_index[1]
            #
            Em = self.dropout(x_dict['n1'])
            Ed = self.dropout(x_dict['n2'])
            y = Em@Ed.t()
            y = y[m_index, d_index].unsqueeze(-1)
        # y = torch.cat((Em, Ed), dim=1)
        # y = self.fc(y)

        return y


    def pro_data(self, x1, x2, edg_index):
        m_index = edg_index[0]
        d_index = edg_index[1]
        Em = torch.index_select(x1, 0, m_index)  # 沿着x1的第0维选出m_index
        Ed = torch.index_select(x2, 0, d_index)
        return Em, Ed


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