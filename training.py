# -*- coding: utf-8 -*-
"""
@author: matep
"""

import torch
import hashNet as Net
from torch.autograd import Variable
# N = batch size
# D_in = input dimension
# H = hidden dimension
# D_out = output dimension
N, D_in, H, D_out = 100, 15, 15, 1
# M = hash table size
M = 1000
#READ INPUTS IN x
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out))
#READ OUTPUTS IN Y

model = Net.HashNet(D_in, H, D_out)

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

for t in range(500):
    
    y_pred = model(x)
    
    loss = criterion(y_pred, y)
    print(t, loss)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()