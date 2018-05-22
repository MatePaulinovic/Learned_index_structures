# -*- coding: utf-8 -*-
"""
@author: matep
"""

import torch
import torch.utils.data
import numpy

class HashDataset(torch.utils.data.dataset.Dataset):
    
    def __init__(self, source=None, M=10e7, cuda=False, listSource=False):
        self.M = M
        if cuda:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor
        
        self.xs = []
        self.ys = []
        if not listSource:
            self.transform = True
            
            with open(source) as f:
                for line in f:
                    parts = line.split(",")
                    self.xs.append(parts[0])
                    self.ys.append(parts[1])
            
        else:
            self.transform = False
            self.xs = []
            self.ys = []
                
            
    def __getitem__(self, index):
        kmer = self.xs[index]
        value = self.ys[index]
        
        if self.transform:
            value = self.transform(value)
        
        #transform to pytorch tensors
        kmer = torch.from_numpy(numpy.asarray(list(map(int, kmer)), dtype=numpy.float32))
        value = torch.from_numpy(numpy.asarray(value, dtype=numpy.float32).reshape([1,1]))
        kmer = kmer.type(self.dtype)
        value = value.type(self.dtype)
        
        return kmer, value
    
    
    def __len__(self):
        return len(self.xs) 
    

    def transform(self, value):
        val = int(value)
        
        return (val % self.M) / self.M
    
    
    def add_item(self, x, y):
        self.xs.append(x)
        self.ys.append(y)
        

class HashCDFDataset(torch.utils.data.dataset.Dataset):

    
    def __init__(self, source=None, targets=None, cuda=False, listSource=False, M=1000000):
        if cuda:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor
        
        self.xs = []
        self.ys = []
        
        if not listSource:
            
            with open(source) as f:
                for line in f:
                    parts = line.split(",")
                    self.xs.append(parts[0])
            
        else:
            self.xs = source
        
        
        if targets is None:
            self.xs.sort()
            self.ys = numpy.random.random_sample(len(self.xs))
            self.ys.sort()
        
        else:
            self.ys = targets
            
        
    def __getitem__(self, index):
        kmer = self.xs[index]
        value = self.ys[index]
      
        #transform to pytorch tensors
        kmer = torch.from_numpy(numpy.asarray(list(map(int, kmer)), dtype=numpy.float32))
        value = torch.from_numpy(numpy.asarray(value, dtype=numpy.float32).reshape([1,1]))
        kmer = kmer.type(self.dtype)
        value = value.type(self.dtype)
        
        return kmer, value
    
    
    def __len__(self):
        return len(self.xs) 
    
