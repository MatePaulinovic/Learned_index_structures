# -*- coding: utf-8 -*-
"""
@author: matep
"""

import torch

class MSELossWithMonotonicityPenalty(torch.nn.Module):
    
    
    def __init__(self, model, lam=0.01, delta=0.001,):
        super(MSELossWithMonotonicityPenalty, self).__init__()
        self.model = model
        self.lam = lam
        self.delta = delta
        
        
    def forward(self, x, y_pred, y):
        mse = torch.mean(((y_pred - y) ** 2)) 
        return mse
        self.model.eval()
        y_delta = self.model(self.increase_tensor(x))
        self.model.train()        
        
        mon_diff = torch.nn.functional.relu(y_pred - y_delta)
        
        n_h = list(torch.nonzero(mon_diff).size())
        if not n_h:
            return torch.sum(mse).mean()
        
        mon_penalty = self.lam * (mon_diff ** 2) / n_h[0]
        #print(torch.sum(mon_penalty + mse).size())
        return torch.sum(mse + mon_penalty)
        
    
    def increase_tensor(self, x):
        incr = torch.torch.transpose(torch.cat((torch.zeros([x.size()[1] - 1, x.size()[0]]), self.delta * torch.ones([1, x.size()[0]]))), 0, 1)
        incr = torch.autograd.Variable(incr)
        
        return x + incr