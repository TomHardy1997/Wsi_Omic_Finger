import torch
import ipdb
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import os

from torch.utils.data import DataLoader, SubsetRandomSampler
import pandas as pd
import torch
import torch.utils.data
import ast

class Response_WSI_Gene_Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_path):
        super(Response_WSI_Gene_Dataset, self).__init__()

        df = pd.read_csv(csv_path, low_memory=False)
        # import ipdb;ipdb.set_trace()
        # df = df[df['measure_of_response'] != 'Partial Response'].reindex()
        # import ipdb;ipdb.set_trace()
        dict = {'Stable Disease': 0, 'Partial Response': 1, 'Complete Response': 2, 'Clinical Progressive Disease':3}
        # dict = {'Stable Disease': 0, 'Complete Response': 1, 'Clinical Progressive Disease':2}
        self.slide = df['slide_id']
        self.label_col = df['measure_of_response'].map(dict)
        # dict = {'Stable Disease': 0, 'Partial Response': 1, 'Complete Response': 2, 'Clinical Progressive Disease':3}
        self.data_dir = '/home/stat-jijianxin/gene/PORPOISE-master/TCGA-GBM/'
        self.num_classes = len(self.label_col.unique())
        self.case_id = df['case_id']
        self.drug = df['finger']
        self.gene = df.iloc[:, 13:]

    def __len__(self):
        return len(self.slide)

    def __getitem__(self, idx):
        slide_id = self.slide[idx]
        label = torch.tensor(self.label_col[idx])
        gene = torch.tensor(self.gene.iloc[idx])
        wsi = os.path.join(self.data_dir, 'pt_files', '{}.pt'.format(self.slide[idx].rstrip('.svs')))
        slide_feature = torch.load(wsi)
        finger = torch.tensor(ast.literal_eval(self.drug[idx]))
        return slide_id, label, gene.unsqueeze(dim=0), slide_feature, finger


if __name__ == '__main__':
    dataset = Response_WSI_Gene_Dataset(csv_path='/home/stat-jijianxin/gene/PORPOISE-master/datasets_csv/tcga_gbmlgg_trian_new_finger_clean.csv.zip')
    # 定义训练集和验证集的比例
    import ipdb;ipdb.set_trace()
    validation_split = 0.2
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    # 随机打乱数据集
    np.random.shuffle(indices)

    # 分割数据集
    train_indices, val_indices = indices[split:], indices[:split]

    # 创建训练集和验证集的数据加载器
    batch_size = 1
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=8)
    import ipdb; ipdb.set_trace()