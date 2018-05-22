# -*- coding: utf-8 -*-
"""
@author: matep
"""

import torch
import hashNet as Net
import hashDataset as Hds
import numpy
import dataVisualisation
import lossFunctions
from torch.autograd import Variable
# N = batch size
# D_in = input dimension
# H = hidden dimension
# D_out = output dimension
N, D_in, H, D_out = 128, 16, 32, 1
# M = hash table size
M = 1e7

dataset = Hds.HashCDFDataset(source="./data/training_set/GRCh37/NT_113909.1.txt", M=M)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=N, shuffle=True, drop_last=False, num_workers=0)
print("Loaded dataset")

#model = Net.HashNet(D_in, H, D_out)
model = Net.ShallowHashNet(D_in, D_in, D_out)

#criterion = torch.nn.MSELoss(size_average=True)
criterion = lossFunctions.MSELossWithMonotonicityPenalty(model)
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.9, momentum=0.1)

#batch_ids = []
#losses = []
model.train()
for i in range(5):
    for batch_idx, (data, target) in enumerate(train_loader):
        #data, target = data.cuda(async=True), target.cuda(async=True) # On GPU
        data, target = Variable(data), Variable(target)
    
        #print("batch_idx: {}".format(batch_idx))
    
        optimizer.zero_grad()
        output = model(data)
        #print("Output: {}, target: {}".format(output, target))
        loss = criterion(data, output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))
        #batch_ids.append(batch_idx * len(data))
        #losses.append(loss.data[0])
        

#torch.save(model.state_dict(), "./hash_model_state_norm.ser")
#dataVisualisation.plot_loss(numpy.asarray(batch_ids, dtype=numpy.float32), numpy.asarray(losses, dtype=numpy.float32))

print("Done")
