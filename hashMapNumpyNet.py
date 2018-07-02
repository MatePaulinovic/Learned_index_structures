# -*- coding: utf-8 -*-
"""
@author: matep
"""

# -*- coding: utf-8 -*-
"""
@author: matep
"""

import torch
import numpy

class HashMapNumpyNet():
    
    
    def __init__(self, net, size=100):
        self.table = [[] for bucket in range(int(size))]
        self.size = int(size)
        self.collisions = 0
        self.filled_slots = 0
        self.model = net
        par = net.layers[0][0].parameters()
        self.inputSize = next(par).size(1)
        
        
    def insert(self, key, value):
        key_index = self.hash_fun(key) % self.size
        #print(type(key_index))
        bucket = self.table[key_index]
        
        if not bucket:
            self.filled_slots += 1
            bucket.append((key, value))
            return 
            
        tmp_bucket = list(filter(lambda t: t[0] != key, bucket))
       
        if len(tmp_bucket) == len(bucket):
            self.collisions += 1
            
        tmp_bucket.append((key, value))
        self.table[key_index] = tmp_bucket
        

    def get(self, key):
        key_index = self.hash_fun(key) % self.size
        bucket = self.table[key_index]
        
        for k,v in bucket:
            if k == key:
                return v
            
        return None
        
    
    def hash_fun(self, key):
        #print("racunam kljuc")
        list_key = list(key)
        x = numpy.asarray(list_key, dtype=numpy.float32)
        #print(x.size()
        #x = self.model.forward(x)
        #print(x)
        #print(type((self.model(x)[0][0]).data))
        #print("PROSAO")
        #print(type(self.model.forward(x)))
        return int(self.model.forward(x) * self.size) % self.size
        
