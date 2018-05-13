# -*- coding: utf-8 -*-
"""
@author: matep
"""


import hashMap
import subprocess
import sys


try:
    filePath = sys.argv[1]
    scaling = 1.0
    if len(sys.argv) == 2:
        scaling = float(sys.argv[2])
        
    f = open(filePath, 'r')
    fileLength = int(subprocess.check_output('wc -l {}'.format(filePath), shell=True).split()[0])
    
    hMap = hashMap.HashMap(size=fileLength*scaling, kmer=False)
    
    
    for line in f.readlines():
        parts = line.split(",")
        hMap.insert(parts[0], parts[0])
    
    f.close()
    print("File length: " + str(fileLength))
    print("Total unique inputs: " + str(hMap.collisions + hMap.filled_slots))
    print("Hash map size: " + str(hMap.size))
    print("Collisions: " + str(hMap.collisions))
    print("Filled slots: " + str(hMap.filled_slots))
    print("Empty slots: " + str(hMap.size - hMap.filled_slots))    
    print("Collison percentage: " + str(hMap.collisions / (hMap.collisions + hMap.filled_slots)))
    
except Exception as e:
    print(str(e))