from models import get_Hdata
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, GCN2Conv, SAGEConv, GATConv, HGTConv, Linear
from torch_geometric.utils import to_undirected
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import roc_auc_score
import os
from utils import get_data
from train_model import CV_train

def set_seed(seed):
    torch.manual_seed(seed)
    #进行随机搜索的这个要注释掉
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class Config:
    def __init__(self):
        self.datapath = './data/'
        self.save_file = './save_file/'

        self.kfold = 5
        self.epochs = 400
        self.print_epoch = 20
        self.maskMDI = False
        self.hidden_channels = 512   # 256 512
        self.num_heads = 4   # 4 8
        self.num_layers = 4   # 4 8
        self.self_encode_len = 256
        self.globel_random = 100
        self.other_args = {'arg_name': [], 'arg_value': []}


def set_attr(config, param_search):
    param_grid = param_search
    param_keys = param_grid.keys()
    param_grid_list = list(ParameterGrid(param_grid))
    for param in param_grid_list:
        config.other_args = {'arg_name': [], 'arg_value': []}
        for keys in param_keys:
            setattr(config, keys, param[keys])
            config.other_args['arg_name'].append(keys)
            print(keys,param[keys])
            config.other_args['arg_value'].append(param[keys])
        yield config
    return 0

if __name__ == '__main__':

    param_search = {
        'hidden_channels': [512],
        'num_heads': [16],
        'num_layers': [6],
        'hidden_bias_len': [32],
    }

    save_file = 'search-h32'

    params_all = Config()
    param_generator = set_attr(params_all, param_search)
    data_list = []
    while True:
        try:
            params = next(param_generator)
        except:
            break
        data_tuple = get_data(file_pair=params.datapath + 'c_d.csv', params=params)
        data_idx, auc_name = CV_train(params, data_tuple)
        data_list.append(data_idx)
    if data_list.__len__() > 1:
        data_all = np.concatenate(tuple(x for x in data_list), axis=0)
    else:
        data_all = data_list[0]
    np.save(params_all.save_file + save_file+'.npy', data_all)
    print(auc_name)
