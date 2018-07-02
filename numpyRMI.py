# -*- coding: utf-8 -*-
"""

@author: matep
"""

import torch
import numpy
import os
import math
import numpyNet

class npRMI():
    
    def __init__(self, head_net, child_nets, M):
        self.M = M
        
        self.layers = []
        self.layers.append([numpyNet.NumpyNet(head_net.parameters())]) #16 32 1
        
        self.layers.append([numpyNet.ShallowNumpyNet(child_nets[i].parameters()) for i in range(len(child_nets))]) #16 16 1
            
        self.layer_sizes = [1] + [len(child_nets)]
        self.num_of_layers = 2
    

    def forward(self, x):
        y_pred = 0
        for i in range(self.num_of_layers):
            expert_num = abs(math.floor(y_pred * self.layer_sizes[i])) % self.layer_sizes[i]
            #print(expert_num)
            y_pred = self.layers[i][expert_num].forward(x).data[0][0]
            #print(type(y_pred))
        return y_pred
    

    
            
