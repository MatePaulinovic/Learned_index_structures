# -*- coding: utf-8 -*-
"""
@author: matep
"""

import hashNet

class RMI():
    
    def __init__(self, numOfModels=[10e5]):
        self.layer1 = [hashNet.HashNet(16, 32, 1)]
        self.layer2 = [hashNet.ShallowHashNet(16,16,1) for i in range(10e5)]
        self.numOfModels = numOfModels        
    
        
        