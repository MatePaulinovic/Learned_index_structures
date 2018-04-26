# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 12:57:09 2018

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
        self.hidden1 = torch.nn.ReLU()
        self.hidden2 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(H, D_out)
        
        
    def forward(self, x):
        h_lin1 = self.linear1(x)
        h_hid1 = self.hidden1(h_lin1)
        h_hid2 = self.hidden2(h_hid1)
        y_pred = self.linear2(h_hid2)
        return y_pred
    
