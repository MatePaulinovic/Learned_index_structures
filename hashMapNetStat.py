# -*- coding: utf-8 -*-
"""
@author: matep
"""

import torch
from hashNet import HashNet
import hashMapNet
import subprocess
import sys


try:
    filePath = sys.argv[1]
    scaling = 1.0
    if len(sys.argv) == 3:
        scaling = float(sys.argv[2])
    
    f = open(filePath, 'r')
    fileLength = int(subprocess.check_output('wc -l {}'.format(filePath), shell=True).split()[0])
    
    D_in, H, D_out = 16, 32, 1
    model = HashNet(D_in, H, D_out)
    model.load_state_dict(torch.load("./hash_model_state_norm.ser"))
 
    hmn = hashMapNet.HashMapNet(net=model, size=fileLength*scaling)
    
    counter = 0
    for line in f.readlines():
        parts = line.split(",")
        hmn.insert(parts[0], parts[0])
        counter += 1
        if counter % 1000 == 0:
            print(counter)
    
    f.close()
    
    print("File length: " + str(fileLength))
    print("Total unique inputs: " + str(hmn.collisions + hmn.filled_slots))
    print("Hash map size: " + str(hmn.size))
    print("Collisions: " + str(hmn.collisions))
    print("Filled slots: " + str(hmn.filled_slots))
    print("Empty slots: " + str(hmn.size - hmn.filled_slots))    
    print("Collison percentage: " + str(hmn.collisions / (hmn.collisions + hmn.filled_slots)))
    
except Exception as e:
    print(str(e))


