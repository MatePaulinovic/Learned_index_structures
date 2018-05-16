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
N, D_in, H, D_out = 128, 16, 32, 1
# M = hash table size
M = 1000

dataset = Hds.HashDataset("./data/training_set/GRCh37/NT_113878.1.txt", M, cuda=True)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=N, shuffle=True, drop_last=False, num_workers=0)
print("Loaded dataset")

model = Net.HashNet(D_in, H, D_out).cuda()

criterion = torch.nn.MSELoss(size_average=True).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)


model.train().cuda()
for batch_idx, (data, target) in enumerate(train_loader):
    #data, target = data.cuda(async=True), target.cuda(async=True) # On GPU
    data, target = Variable(data), Variable(target)
    
    #print("batch_idx: {}".format(batch_idx))
    
    optimizer.zero_grad()
    output = model(data)
    #print("Output: {}, target: {}".format(output, target))
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % 1000 == 0:
        print('Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
