# -*- coding: utf-8 -*-
"""
@author: matep
"""

import torch
from hashNet import HashNet, ShallowHashNet
from numpyNet import NumpyNet, ShallowNumpyNet
import hashMapNet
import RMI
import subprocess
import sys
import numpyRMI
import hashMapNumpynet


try:
    filePath = sys.argv[1]
    scaling = 1.0
    if len(sys.argv) == 3:
        scaling = float(sys.argv[2])
    
    f = open(filePath, 'r')
    fileLength = int(subprocess.check_output('wc -l {}'.format(filePath), shell=True).split()[0])
    
    M = 1e7    
    D_in, H, D_out = 16, 32, 1
    #stages = [int(1e5)]
    stages = [int(1e4)]
    head_net_params = [D_in, H, D_out]
    child_net_params = [D_in, D_in, D_out]
    rmi = RMI.RMI(HashNet, head_net_params, ShallowHashNet, child_net_params, stages, M)
    #rmi.load("./data/serialization/")
    rmi.load("./data/serialization/4k")
    print("Loaded")
    rmi.evaluation()
    npRMI = numpyRMI.npRMI(rmi.layers[0][0], rmi.layers[1], M)
    hmn = hashMapNumpynet.HashMapNumpyNet(npRMI, fileLength*scaling)
    #hmn = hashMapNet.HashMapNet(net=rmi, size=fileLength*scaling)
    print("Start inserting")
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
