import numpy as np
from sklearn import metrics
from models import get_Hdata
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, GCN2Conv, SAGEConv, GATConv, HGTConv, Linear
from torch_geometric.utils import to_undirected
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os

def get_metrics(real_score, predict_score):
    real_score, predict_score = real_score.flatten(), predict_score.flatten()
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num*np.arange(1, 1000)/1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN

    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]

    # np.savetxt(roc_path.format(i), ROC_dot_matrix)

    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])

    recall_list = tpr
    precision_list = TP/(TP+FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]

    # np.savetxt(pr_path.format(i), PR_dot_matrix)

    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])

    f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
    accuracy_list = (TP+TN)/len(real_score.T)
    specificity_list = TN/(TN+FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    print( ' auc:{:.4f} ,aupr:{:.4f},f1_score:{:.4f}, accuracy:{:.4f}, recall:{:.4f}, specificity:{:.4f}, precision:{:.4f}'.format( auc[0, 0],aupr[0, 0], f1_score, accuracy, recall, specificity, precision))
    return [real_score, predict_score, auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision], \
           ['y_true','y_score','auc', 'prc', 'f1_score', 'acc', 'recall', 'specificity', 'precision']


def get_data(file_pair, params):
    adj_matrix = pd.read_csv(file_pair.md, header=None, index_col=None).values

    # 将邻接矩阵转换为张量，这里要将列索引加上总节点数，使得列节点的索引不与行节点重复
    edge_index_pos = np.column_stack(np.argwhere(adj_matrix != 0))
    edge_index_pos = torch.tensor(edge_index_pos, dtype=torch.long)

    # 获取不相关连接 为了构建训练数据
    edge_index_neg = np.column_stack(np.argwhere(adj_matrix == 0))
    edge_index_neg = torch.tensor(edge_index_neg, dtype=torch.long)

    ctrl = True
    if params.ablation == 'no-rand':
        x1 = []
        x2 = []
        for i in range(0, len(file_pair.mm)):
            x1.append(pd.read_csv(file_pair.mm[i], header=None, index_col=None).values)
            x2.append(pd.read_csv(file_pair.dd[i], header=None, index_col=None).values)

        x1 = torch.tensor(np.array(x1).mean(0), dtype=torch.float32)
        x2 = torch.tensor(np.array(x2).mean(0), dtype=torch.float32)
    else:
        # 创建数据
        x1 = torch.randn((adj_matrix.shape[0], 128))  # 没有节点特征，所以设置为1 m
        x2 = torch.randn((adj_matrix.shape[1], 128))  # 没有节点特征，所以设置为1 d

        x1 = torch.randn((adj_matrix.shape[0], params.self_encode_len))  # 没有节点特征，所以设置为1 m
        x2 = torch.randn((adj_matrix.shape[1], params.self_encode_len))  # 没有节点特征，所以设置为1 d


    num_pos_edges_number = edge_index_pos.shape[1]
    if params.ablation == '1-5':
        num_pos_edges_number = num_pos_edges_number * 5
    if params.ablation == '1-10':
        num_pos_edges_number = num_pos_edges_number * 10
    selected_neg_edge_indices = torch.randint(high=edge_index_neg.shape[1], size=(num_pos_edges_number,),
                                               dtype=torch.long)
    edge_index_neg_selected = edge_index_neg[:, selected_neg_edge_indices]
    edg_index_all = torch.cat((edge_index_pos, edge_index_neg_selected), dim=1)
    y = torch.cat((torch.ones((edge_index_pos.shape[1], 1)),
                   torch.zeros((edge_index_neg_selected.shape[1], 1))), dim=0)  # 将所有y值设置为1,0
    # 获取平衡样本

    data = get_Hdata(x1, x2, edge_index_pos)

    return data, y, edg_index_all
