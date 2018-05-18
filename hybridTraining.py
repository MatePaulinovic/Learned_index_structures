# -*- coding: utf-8 -*-
"""
@author: matep
"""

import torch
import hashNet as Net
import hashDataset as Hds
import numpy
import dataVisualisation
from torch.autograd import Variable



def train(model, criterion, optimizer, dataloader, epochs=1, visualize=False):
    batch_ids = []
    losses = []
    model.train()
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
            batch_ids.append(batch_idx)
            losses.append(loss)
    


# N = batch size
# D_in = input dimension
# H = hidden dimension
# D_out = output dimension
N, D_in, H, D_out = 128, 16, 32, 1
# M = hash table sizenao
M = 10e7

dataset = Hds.HashDataset("./data/training_set/training.txt", M)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=N, shuffle=True, drop_last=False, num_workers=0)
print("Loaded dataset")
    
stages = [1,10e5]

models = [Net.HashNet(D_in, H, D_out), [ Net.ShallowHashNet(D_in, D_in, D_out) for i in range(10e5)]]
dataSets = [dataset, [Hds.HashDataset(None, M, listSource=True)]]

for i in range(0, 2):
    for j, model in enumerate(models[i]):
        criterion = torch.nn.MSELoss(size_average=True)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-2, alpha=0.9, momentum=0.1)
       
        model.train()
        
        if i == 0:
            train(model, criterion, optimizer, train_loader)
            model.eval()
            for (data, target) in train_loader:
                p = model(Variable(data)).data[0] / stages[i+1]  
                dataSets[i + 1][p].add_item(data, target)
        
        else:
            dataloader = torch.utils.data.DataLoader(dataSets[i][j]) 
            train(model, criterion, optimizer, dataloader, epochs=100)  

torch.save(model.state_dict(), "./hash_model_state_norm.ser")

