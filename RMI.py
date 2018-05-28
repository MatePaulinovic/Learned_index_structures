# -*- coding: utf-8 -*-
"""
@author: matep
"""

import torch
import os
import math

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
            expert_num = abs(math.floor(y_pred * self.layer_sizes[i])) % self.layer_sizes[i]
            #print(expert_num)
            y_pred = self.layers[i][expert_num].forward(x).data[0][0]
            #print(type(y_pred))
        return y_pred
    

    def evaluation(self):
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
                
                #print("Current filename: " + str(filename))         
                parts = (os.path.splitext(filename)[0]).split("_")
                j = int(parts[len(parts) - 1])
                i = int(parts[len(parts) - 2])
                self.layers[i][j].load_state_dict(torch.load(os.path.realpath(dir_name + filename)))
        except Exception as ex:
            print(ex)
            print("Specified wrong directory or model structure")
            
            
