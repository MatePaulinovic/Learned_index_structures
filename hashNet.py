# -*- coding: utf-8 -*-
"""
@author: matep
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class HashNet(nn.Module):
    
    def __init__(self, D_in, H, D_out):
        super(HashNet, self).__init__()
        
        self.linear1 = nn.Linear(D_in, H)
        self.hidden1 = nn.Linear(H, H)
        self.hidden2 = nn.Linear(H,H)
        self.linear2 = nn.Linear(H, D_out)
        #self.bn1 = nn.BatchNorm1d(H)
        #self.bn2 = nn.BatchNorm1d(H)
        #self.bn3 = nn.BatchNorm1d(H)
        
    def forward(self, x):
        h_lin1 = F.relu(self.linear1(x))
        #h_lin1 = self.bn1(h_lin1)
        h_hid1 = F.relu(self.hidden1(h_lin1))
        #h_hid1 = self.bn2(h_hid1)
        h_hid2 = F.relu(self.hidden2(h_hid1))
        #h_hid2 = self.bn3(h_hid2)
        y_pred = F.sigmoid(self.linear2(h_hid2))
        return y_pred
    
    
class ShallowHashNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(ShallowHashNet, self).__init__()
        
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
        
    
    def forward(self, x):
        h_lin1 = self.linear1(x)
        y_pred = self.linear2(h_lin1)
        return y_pred
