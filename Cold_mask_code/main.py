import multiprocessing

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
from utils import get_data
from train_mask_model import CV_train

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
        self.datapath = './data/'  # Path to the data folder
        self.save_file = './save_file3/'  # Path to the folder where results are saved

        self.kfold = 5  # Number of folds for cross-validation
        self.repeat = 100  # Number of times to repeat the experiment
        self.epochs = 500  # Number of training epochs
        self.each_epochs_print = self.print_epoch = 10  # Print progress every 10 epochs
        self.Tsen = True  # Some toggle for an unknown feature
        self.maskMDI = False  # Toggle for masking in some context
        self.hidden_channels = 512  # Size of the hidden layers
        self.num_heads = 16  # Number of attention heads in the model
        self.num_layers = 6  # Number of layers in the model
        self.self_encode_len = 64  # Length of the self-encoding
        self.globel_random = 100  # Global random seed
        self.other_args = {'arg_name': [], 'arg_value': []}  # Placeholder for additional arguments
        # Various ablation study options:
        self.ablation = 'no-rand'  # 'no-rand' (use self), 'no-HGT' (use HAN),
                                   # 'no-catmul' (use fully connected),
                                   # 'no-H4' (use the last layer), 'no-HNN' (use GCN)

class Data_paths:
    def __init__(self, md_file='c_d.csv'):  # Default to 'c_d.csv' (data1), can switch to 'c_d2.csv' (data2)
        self.paths = './data/'  # Path to the data folder
        self.md = self.paths + md_file  # Full path to the main data file
        # Uncomment these lines if other data files are needed:
        # self.mm = [self.paths + 'm_gs.csv', self.paths + 'm_ss.csv']
        # self.dd = [self.paths + 'm_gs.csv', self.paths + 'm_ss.csv']

def m_train(ab):
    print(f'Starting: {ab}')  # Print the ablation option being used
    params = Config()  # Initialize configuration parameters
    file_dir = params.save_file  # Get the directory for saving results
    for sp in ['drug', 'gene', 'drug_gene']:  # Loop through these three categories
        params.ablation = ab  # Set the ablation type
        data_tuple = get_data(file_pair=Data_paths(), params=params)  # Get the data based on the current configuration
        params.save_file = file_dir + f'/{sp}/'  # Update the save directory for each category
        if not os.path.exists(params.save_file):
            os.makedirs(params.save_file, exist_ok=True)  # Create the directory if it doesn't exist
        params.sp = sp  # Set the current category (drug, gene, or drug_gene)
        CV_train(ab, params, data_tuple, repeat=params.repeat)  # Train the model using cross-validation
        print(f'Completed: {ab}-{sp}\n')  # Print completion message for the current category


if __name__ == '__main__':

    abla = ['all',]
    for a in abla:
        m_train(a)

    # multiprocessing.set_start_method('spawn')
    #
    # pool = multiprocessing.Pool(processes=abla.__len__())  # 这里设置为4个进程，你可以根据需要调整
    #
    # items_to_process = abla  # 你要处理的项目列表
    #
    # # 使用进程池来并行执行 worker_function
    # pool.map(m_train, items_to_process)
    #
    # # 关闭进程池，确保所有进程都被终止
    # pool.close()
    # pool.join()

