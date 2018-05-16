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
        
        self.linear1 = torch.nn.Linear(D_in, H)
        self.hidden1 = torch.nn.Linear(H, H)
        self.hidden2 = torch.nn.Linear(H,H)
        self.linear2 = torch.nn.Linear(H, D_out)
        #self.linear2 = torch.nn.Sigmoid()

        
    def forward(self, x):
        h_lin1 = F.relu(self.linear1(x))
        h_hid1 = F.relu(self.hidden1(h_lin1))
        h_hid2 = F.relu(self.hidden2(h_hid1))
        y_pred = self.linear2(h_hid2)
        return y_pred
    
    
class ShallowHashNet(nn.Module):
    def __init_(self, D_in, H, D_out):
        super(ShallowHashNet, self).__init__()
        
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
        
    
    def forward(self, x):
        h_lin1 = self.linear1(x)
        y_pred = self.linear2(h_lin1)
        return y_pred
