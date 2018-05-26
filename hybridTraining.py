# -*- coding: utf-8 -*-
"""
@author: matep
"""

import math
import torch
import hashNet as Net
import hashDataset as Hds
import RMI
import lossFunctions
from torch.autograd import Variable



def train(model, criterion, optimizer, dataloader, epochs): #visualize=False):
    #batch_ids = []
    #losses = []
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #data, target = data.cuda(async=True), target.cuda(async=True) # On GPU
        data, target = Variable(data), Variable(target)
    
        #print("batch_idx: {}".format(batch_idx))
    
        optimizer.zero_grad()
        output = model(data)
        #print("Output: {}, target: {}".format(output, target))
        loss = criterion(output, target)
        #loss = criterion(data, output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))
            #batch_ids.append(batch_idx)
          #  losses.append(loss)
    

    
# N = batch size
# D_in = input dimension
# H = hidden dimension
# D_out = output dimension
N, D_in, H, D_out = 128, 16, 32, 1
# M = hash table sizenao
M = int(1e7)

dataset = Hds.HashCDFDataset(source="./data/training_set/training.txt", M=M)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=N, shuffle=True, drop_last=False, num_workers=0)
print("Loaded dataset")
    
stages = [int(1e5)]
head_net_params = [D_in, H, D_out]
child_net_params = [D_in, D_in, D_out]
rmi = RMI.RMI(Net.HashNet, head_net_params, Net.ShallowHashNet, child_net_params, stages, M)

print("Stvorio RMI")

data_dist = []
data_dist.append([[]])
for stage in stages:
    data_dist.append([ [] for i in range(stage)])
    
data_dist[0][0] = ([i for i in range(0,len(dataset))])

print("Stvorio data_dist")

rmi.train()

for i in range(rmi.num_of_layers):

    for j, model in enumerate(rmi.layers[i]):
        #print("VELIČINA ELEMENATA NA OVOJ STAGEU " + str(len(rmi.layers[i])))
        print("Treniram model {} na raznini {}".format(j, i))
        criterion = torch.nn.MSELoss(size_average=True)
        #criterion = lossFunctions.MSELossWithMonotonicityPenalty(model)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-2, alpha=0.9, momentum=0.1)
        
        if len(data_dist[i][j]) == 0:
            continue
        
        inputs = []
        outputs = []
        for valid_index in data_dist[i][j]:
            inputs.append(dataset.xs[valid_index])
            outputs.append(dataset.ys[valid_index])
            
        tmp_dataset = Hds.HashCDFDataset(source=inputs, targets=outputs, listSource=True, M=M)
        train_loader = torch.utils.data.DataLoader(dataset=tmp_dataset, batch_size=N, shuffle=True, drop_last=False, num_workers=0)  
        
        train(model, criterion, optimizer, train_loader, 1000)
        
        model.eval()
        if i == rmi.num_of_layers - 1:
            continue
        
        for index in data_dist[i][j]:
            data, _ = dataset[index]
            data = data.unsqueeze(0)       
            p = (model(Variable(data)).data[0][0] * rmi.layer_sizes[i+1]) 
            data_dist[i + 1][math.floor(p)].append(index)
        
print("Zapisivanje")

"""
for data in data_dist:
    for i, d in enumerate(data):
        if len(d) == 0:
            continue
        print("{}. {}".format(i, d))
"""
rmi.save("./data/serialization/rmi")





















