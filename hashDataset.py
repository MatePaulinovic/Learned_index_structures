# -*- coding: utf-8 -*-
"""
@author: matep
"""

import torch
import torch.utils.data
import numpy

class HashDataset(torch.utils.data.dataset.Dataset):
    
    
    __xs = []
    __ys = []
    
    
    def __init__(self, source, M=1000):
        self.M = M
            
        with open(source) as f:
            for line in f:
                parts = line.split(",")
                self.__xs.append(parts[0])
                self.__ys.append(parts[1])
                
            
    def __getitem__(self, index):
        kmer = self.__xs[index]
        value = self.__ys[index]
        
        if self.transform is not None:
            value = self.transform(value)
        
        #transform to pytorch tensors
        kmer = torch.from_numpy(numpy.asarray(list(map(int, kmer)), dtype=numpy.float32))
        value = torch.from_numpy(numpy.asarray(value, dtype=numpy.float32).reshape([1,1]))
        
        return kmer, value
    
    
    def __len__(self):
        return len(self.__xs) 
    

    def transform(self, value):
        val = int(value)
        
        return (val % self.M) / self.M
    