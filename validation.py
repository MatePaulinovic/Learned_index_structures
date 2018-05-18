# -*- coding: utf-8 -*-
"""
@author: matep
"""

import torch
import sys
import subprocess
import hashDataset
import hashNet
from torch.autograd import Variable

try:
    filePath = sys.argv[1]
    scaling = 1.0
    if len(sys.argv) == 3:
        scaling = float(sys.argv[2])
    
    f = open(filePath, 'r')
    fileLength = int(subprocess.check_output('wc -l {}'.format(filePath), shell=True).split()[0])
    
    D_in, H, D_out = 16, 32, 1
    model = hashNet.HashNet(D_in, H, D_out)
    model.load_state_dict(torch.load("./hash_model_state_norm.ser"))
    model.eval()
    
    dataset = hashDataset.HashDataset(source="./data/training_set/validation.txt", M=10e7 )
    dataloader = torch.utils.data.DataLoader(dataset)

    counter = 0
    for i, (data, target) in enumerate(dataloader):
        y_pred = model(Variable(data)).data[0]
        if y_pred != target:
            print("{}.\ty={}\ty_pred={}".format(i, target, data ))
            counter += 1
    
    
    print("Misses: " + str(counter))
    print("Accuracy: " + str(counter / dataset.__len__()))
   
except Exception as e:
    print(str(e))
