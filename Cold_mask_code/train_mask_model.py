import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, GCN2Conv, SAGEConv, GATConv, HGTConv, Linear
from torch_geometric.utils import to_undirected
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from models import HGT
from globel_args import device
from utils import get_metrics
from sklearn.model_selection import KFold
import copy
import os,sys

def trian_model(data,y, edg_index_all, train_idx, test_idx, param):
    hidden_channels, num_heads, num_layers = (
        param.hidden_channels, param.num_heads, param.num_layers,
    )

    epoch_param = param.epochs

    model = HGT(hidden_channels, num_heads=num_heads, num_layers=num_layers, data=data,params=param).to(device)
    model.params = param
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0002)
    data_temp = copy.deepcopy(data)

    # Mask Data
    data_temp[('n1', 'e1', 'n2')].edge_index = data[('n1', 'e1', 'n2')].edge_index[:, train_idx[y.reshape((-1,)) == 1]]
    data_temp[('n2', 'e1', 'n1')].edge_index = data[('n2', 'e1', 'n1')].edge_index[:, train_idx[y.reshape((-1,)) == 1]]

    data_temp['x_dict'] = {ntype: data[ntype].x for ntype in data.node_types}
    edge_index_dict = {}
    for etype in data_temp.edge_types:
        edge_index_dict[etype] = data_temp[etype].edge_index

    data_temp['edge_dict'] = edge_index_dict
    auc_list = []
    model.train()
    for epoch in range(1, epoch_param+1):
        optimizer.zero_grad()
        model.epoch = epoch
        out = model(data_temp['x_dict'], data_temp['edge_dict'],
                    edge_index=edg_index_all.to(device))
        # y_one_hot = F.one_hot(y[train_idx].long().squeeze(), num_classes=2).float().to(device)
        loss = F.binary_cross_entropy_with_logits(out[train_idx].to(device), y[train_idx].to(device))
        loss.backward()
        optimizer.step()
        loss = loss.item()
        if epoch % param.print_epoch == 0:
            model.epoch = 0
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            # 模型验证
            model.eval()
            with torch.no_grad():
                out = model(data_temp['x_dict'], data_temp['edge_dict'],
                            edge_index=edg_index_all)
                out_pred_s = out[test_idx].to('cpu').detach().numpy()
                out_pred = out_pred_s
                y_true = y[test_idx].to('cpu').detach().numpy()
                auc = roc_auc_score(y_true, out_pred)
                print('AUC:', auc)

                auc_idx, auc_name = get_metrics(y_true, out_pred)
                auc_idx.extend(param.other_args['arg_value'])
                auc_idx.append(epoch)
            auc_list.append(auc_idx)
            model.train()
    auc_name.extend(param.other_args['arg_name'])
    return auc_list, auc_name

def mask_func(edg_index_all, mask_sp='drug', test_rate=0.2, test_numb=None):
    edg_index_copy = copy.deepcopy(edg_index_all)
    if mask_sp == 'drug':
        drug_set = set(edg_index_copy[0].tolist())
        if test_numb != None:
            test_numb = test_numb
        else:
            test_numb = int(len(drug_set) * test_rate)

        test_drug = np.random.choice(list(drug_set), size=test_numb, replace=False)
        test_idx = np.isin(edg_index_copy[0].cpu().numpy(), test_drug)
        train_idx = ~test_idx

    elif mask_sp == 'gene':
        gene_set = set(edg_index_copy[1].tolist())
        if test_numb != None:
            test_numb = test_numb
        else:
            test_numb = int(len(gene_set) * test_rate)
        test_gene = np.random.choice(list(gene_set), size=test_numb, replace=False)
        test_idx = np.isin(edg_index_copy[1].cpu().numpy(), test_gene)
        train_idx = ~test_idx

    elif mask_sp == 'drug_gene':

        drug_set = set(edg_index_copy[0].tolist())
        if test_numb != None:
            test_numb = test_numb
        else:
            test_numb = int(len(drug_set) * test_rate/2)

        test_drug = np.random.choice(list(drug_set), size=test_numb, replace=False)
        test_idx1 = np.isin(edg_index_copy[0].cpu().numpy(), test_drug)

        gene_set = set(edg_index_copy[1].tolist())
        if test_numb != None:
            test_numb = test_numb
        else:
            test_numb = int(len(gene_set) * test_rate/2)

        test_gene = np.random.choice(list(gene_set), size=test_numb, replace=False)
        test_idx2 = np.isin(edg_index_copy[1].cpu().numpy(), test_gene)
        test_idx = test_idx1 | test_idx2
        train_idx = ~test_idx
    else:
        raise ValueError("Invalid mask_sp value. Should be 'drug' or 'gene'.")

    return train_idx, test_idx

def CV_train(ab, param, args_tuple=(), repeat=100):
    data, y, edg_index_all = args_tuple
    idx = np.arange(y.shape[0])
    k_number = 1
    mask_sp = param.sp

    test_rate = 0.2
    if mask_sp == 'drug':
        drug_set = set(edg_index_all[0].tolist())
        if repeat != None:
            repeat = repeat
        else:
            repeat = int(len(drug_set) * test_rate)

    elif mask_sp == 'gene':
        gene_set = set(edg_index_all[1].tolist())
        if repeat != None:
            repeat = repeat
        else:
            repeat = int(len(gene_set) * test_rate)


    elif mask_sp == 'drug_gene':

        drug_set = set(edg_index_all[0].tolist())
        if repeat != None:
            repeat1 = repeat
        else:
            repeat1 = int(len(drug_set) * test_rate)

        gene_set = set(edg_index_all[1].tolist())
        if repeat != None:
            repeat2 = repeat
        else:
            repeat2 = int(len(gene_set) * test_rate)

        repeat = int((repeat1 + repeat2)/2)
    kf_auc_list = []

    for r in range(repeat):
        print(f'Running repeat experiment {r+1} out of {repeat}...')
        while True:
            train_idx, test_idx = mask_func(edg_index_all, mask_sp=param.sp, test_numb=1)
            try:
                auc_idx, auc_name = trian_model(data, y, edg_index_all, train_idx, test_idx, param)
                k_number += 1
                kf_auc_list.append(auc_idx)
                break
            except:
                continue

    data_idx = np.array(kf_auc_list)

    np.save(param.save_file + f'{ab}_data_idx.npy', data_idx)
    return data_idx, auc_name


