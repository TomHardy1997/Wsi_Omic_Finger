import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from os.path import join
from collections import OrderedDict

import torch
import torch.nn as nn

class TrilinearFusion(nn.Module):
    def __init__(self, skip=0, use_bilinear=0, gate1=1, gate2=1, dim1=128, dim2=128, scale_dim1=1, scale_dim2=1, mmhid=256, dropout_rate=0.25, extra_dim=64):
        super(TrilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1//scale_dim1, dim2//scale_dim2
        skip_dim = dim1_og+dim2_og if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), 256), nn.ReLU())
        self.encoder2 = nn.Sequential(nn.Linear(256+skip_dim+extra_dim, mmhid), nn.ReLU())

        self.linear_extra = nn.Sequential(nn.Linear(extra_dim, extra_dim), nn.ReLU())

    def forward(self, vec1, vec2, extra_input):
        device = vec1.device  # 获取输入张量的设备

        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)
        else:
            h1 = self.linear_h1(vec1)
            o1 = self.linear_o1(h1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2) * h2)
        else:
            h2 = self.linear_h2(vec2)
            o2 = self.linear_o2(h2)

        o1 = torch.cat((o1, torch.tensor(1, device=device).expand(o1.shape[0], 1)), 1)
        o2 = torch.cat((o2, torch.tensor(1, device=device).expand(o2.shape[0], 1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)

        if self.skip:
            out = torch.cat((out, vec1, vec2), 1)

        extra_out = self.linear_extra(extra_input)
        out = torch.cat((out, extra_out), 1)
        out = self.encoder2(out)

        return out

'''
四个参数
L：输入特征维数
D：隐藏层维数
dropout：正则化
n_classes
'''
class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated,self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()
        ]
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))
        
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)
    
    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A,x

def SNN_Block(dim1, dim2, dropout=0.25):
    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))


def MLP_Block(dim1, dim2, dropout=0.25):
    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            nn.Dropout(p=dropout, inplace=False))    


class Porpoise_MMF(nn.Module):
    def __init__(self, 
                 omic_input_dim, 
                 path_input_dim=1024, 
                 finger_input_dim=1024,
                 dropout=0.25, 
                 n_classes=4, 
                 scale_dim1=8,
                 scale_dim2=8,
                 gate_path=1,
                 gate_omic=1,
                 skip=True,
                 dropinput=0.10,
                 size_arg = 'small'):
        super(Porpoise_MMF, self).__init__()
        self.size_dict_path = {'small':[path_input_dim, 512, 256], 'big':[1024, 512, 384]}
        self.size_dict_omic = {'small':[256, 256]}
        self.size_dict_finger = {'small':[256, 256]}
        self.n_classes = n_classes
        size = self.size_dict_path[size_arg]
        ###正则化
        if dropinput:
            fc = [nn.Dropout(dropinput), nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropinput)]
        else:
            fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.rho = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        hidden = self.size_dict_omic['small']
        fc_omic = [SNN_Block(dim1=omic_input_dim,dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
        self.fc_omic = nn.Sequential(*fc_omic)
        fc_finger = [SNN_Block(dim1=finger_input_dim,dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_finger.append(MLP_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
        self.fc_finger = nn.Sequential(*fc_finger)
        #开始混合
        self.mm = TrilinearFusion(dim1=256, dim2=256, scale_dim1=scale_dim1, gate1=gate_path, scale_dim2=scale_dim2, gate2=gate_omic, skip=skip, mmhid=256,dropout_rate=0.25, extra_dim=256)
        self.classifier_mm = nn.Linear(size[2], n_classes)
    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')
            self.fc_omic = self.fc_omic.to(device)
            self.mm = self.mm.to(device)

        self.rho = self.rho.to(device)
        self.classifier_mm = self.classifier_mm.to(device)
    
    def forward(self,**kwargs):
        x_path = kwargs['x_path']
        # import ipdb;ipdb.set_trace()
        A, h_path = self.attention_net(x_path)
        A = torch.transpose(A, 1, 0)
        A_raw = A
        A = F.softmax(A, dim=1)
        h_path = torch.mm(A, h_path)
        h_path = self.rho(h_path)

        x_omic = kwargs['x_omic']
        h_omic = self.fc_omic(x_omic)
        x_finger = kwargs['x_finger']
        h_finger = self.fc_finger(x_finger)
        h_mm = self.mm(h_path, h_omic, h_finger)
        h_mm = self.classifier_mm(h_mm)
        assert len(h_mm.shape) == 2 and h_mm.shape[1] == self.n_classes

        return h_mm