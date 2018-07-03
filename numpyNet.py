# -*- coding: utf-8 -*-
"""
@author: matep
"""

import numpy as np

def relu(x):
    #np.maximum(x, 0, x)
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
                if len(param.data.numpy().shape) == 1:
                    self.biases.append(np.expand_dims(param.data.numpy(), axis=1))
                else:
                    self.biases.append(param.data.numpy())
        
        
    def forward(self, x):
        x_p = x
        for i in range(len(self.weights) - 1):
            #print("Should get ", np.dot(self.weights[i], x_p).shape)
            x_p = np.dot(self.weights[i],x_p) + self.biases[i]
            #print("Get ", x_p.shape)
            relu(x_p)
            #print("weight ", i, self.weights[i].shape)
            #print("bias ", i, self.biases[i].shape)
            #print(len(self.biases[i].shape))
         
  
        #print(x_p.shape)
        #print(self.weights[len(self.weights) - 1].shape)
        #print(np.dot(self.weights[len(self.weights) - 1], x_p).shape)
        return sigmoid(np.dot(self.weights[len(self.weights) - 1], x_p) + self.biases[len(self.weights) - 1])
    
    
class ShallowNumpyNet:
    
    def __init__(self, params):
        self.weights = []
        self.biases = []
        
        for i, param in enumerate(params):
            if i % 2 == 0:
                self.weights.append(param.data.numpy())
            
            else:
                if len(param.data.numpy().shape) == 1:
                    self.biases.append(np.expand_dims(param.data.numpy(),axis= 1))
                else:
                    self.biases.append(param.data.numpy())        
        
    def forward(self, x):
        x_p = x
        x_p = np.dot(self.weights[0], x_p) + self.biases[0]
            
        return np.dot(self.weights[1], x_p) + self.biases[1]
    
       
