# -*- coding: utf-8 -*-
"""
@author: matep
"""

import torch
import os

class RMI():
    
    def __init__(self, head_net, head_net_params, child_net, child_net_params, layer_sizes, M):
        self.M = M
        
        self.layers = []
        self.layers.append([head_net(head_net_params[0], head_net_params[1], head_net_params[2])]) #16 32 1
        
        for layer_size in layer_sizes:
            self.layers.append([child_net(child_net_params[0], child_net_params[1], child_net_params[2]) for i in range(layer_size)]) #16 16 1
            
        self.layer_sizes = [1] + layer_sizes 
        self.num_of_layers = len(layer_sizes) + 1
    

    def forward(self, x):
        y_pred = 0
        for i in range(self.num_of_layers):
            expert_num = y_pred * self.layer_sizes[i]
            y_pred = self.layers[0][expert_num].forward(x)
        
        return y_pred
    

    def eval(self):
        for i in range(self.num_of_layers):
            for model in self.layers[i]:
                model.eval()
                

    def train(self):
        for i in range(self.num_of_layers):
            for model in self.layers[i]:
                model.train()
                
        
    def save(self, file_start):
        i_len = len(str(len(self.layer_sizes)))
        j_len = max(map(lambda x: len(str(x)), self.layer_sizes))
        for i, layer in enumerate(self.layers):
            for j, model in enumerate(layer):
                file_name = "{}_{:0>{i_pad}}_{:0>{j_pad}}.pth".format(file_start, i, j, i_pad=i_len, j_pad=j_len)
                torch.save(model.state_dict(), file_name)
        
    
    def load(self, dir_name):
        try:
            for filename in os.listdir(dir_name):
                if not filename.endswith('.pth'):
                    continue
            
                parts = (os.path.splitext(filename)[0]).split("_")
                i = parts[len(parts) - 1]
                j = parts[len(parts) - 2]
            
                self.layers[i][j].load_state_dict(torch.load(filename))
    
        except Exception as ex:
            print("Specified wrong directory or model structure")
            
            