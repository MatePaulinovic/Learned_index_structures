# -*- coding: utf-8 -*-
"""
@author: matep
"""

import torch
import hashNet as Net
import hashDataset as Hds
from torch.autograd import Variable
# N = batch size
# D_in = input dimension
# H = hidden dimension
# D_out = output dimension
N, D_in, H, D_out = 100, 15, 15, 1
# M = hash table size
M = 1000
#READ INPUTS IN x
#
dataset = Hds.HashDataset("./data/training_set/NT_113878.1.txt", 1000)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=N, shuffle=True, num_workers=4)
print("Loaded dataset")

#x = Variable(torch.randn(N, D_in))
#y = Variable(torch.randn(N, D_out))
#READ OUTPUTS IN Y

model = Net.HashNet(D_in, H, D_out)

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


model.train()
print("Idem u petlju")
for batch_idx, (data, target) in enumerate(train_loader):
    #data, target = data.cuda(async=True), target.cuda(async=True) # On GPU
    print("USAO U PETLJU")
    data, target = Variable(data), Variable(target)
    
    print("batch_idx: {}".format(batch_idx))
    
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % 10 == 0:
        print('Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

"""
for t in range(500):
    
    y_pred = model(x)
    
    loss = criterion(y_pred, y)
    print(t, loss)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
"""
    
print("Done")