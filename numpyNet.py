# -*- coding: utf-8 -*-
"""
@author: matep
"""

import numpy as np

def relu(x):
    x[x<0] =0
    return

def sigmoid(x):
    return 1/(1 + np.exp(-x))


class NumpyNet:
    
    def __init__(self, params):
        self.weights = []
        self.biases = []
        
        for i, param in enumerate(params):
            if i % 2 == 0:
                self.weights.append(param.data.numpy())
            
            else:
                self.biases.append(param.data.numpy())
        
        
    def forward(self, x):
        x_p = x
        for i in range(len(self.weights) - 1):
            relu(np.dot(x_p, self.weights[i]) + self.biases[i])
            
        return sigmoid(np.dot(x_p, self.weights[len(self.weights) - 1]) + self.biases[len(self.weights) - 1])
    
    
class ShallowNumpyNet:
    
    def __init__(self, params):
        self.weights = []
        self.biases = []
        
        for i, param in enumerate(params):
            if i % 2 == 0:
                self.weights.append(param.data.numpy())
            
            else:
                self.biases.append(param.data.numpy())
        
        
    def forward(self, x):
        x_p = x
        x_p = np.dot(x_p, self.weights[0] + self.biases[0])
            
        return sigmoid(np.dot(x_p, self.weights[1]) + self.biases[1])
    
       