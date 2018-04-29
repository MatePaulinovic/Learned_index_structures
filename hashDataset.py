# -*- coding: utf-8 -*-
"""
@author: matep
"""

import torch
import numpy

class HashDataset(torch.utils.data.Dataset):
    
    
    __xs = []
    __ys = []
    
    
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        
        with open(root_dir + "data.txt") as f:
            for line in f:
                parts = line.split("\t")
                self.__xs.append(parts[0])
                self.__ys.append(parts[1])
                
            
    def __getitem__(self, index):
        kmer = self.__xs[index]
        value = self.__ys[index]
        
        if self.transform is not None:
            value = self.transform(value)
        
        #transform to pytorch tensors
        kmer = torch.from_numpy(numpy.asarray(list(map(int, kmer))))
        value = torch.from_numpy(numpy.asarray(value).reshape([1,1]))
        
        return kmer, value