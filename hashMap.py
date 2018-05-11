# -*- coding: utf-8 -*-
"""
@author: matep
"""

class HashMap:
    
    hash_values = {
                'A' : 0,
                'C' : 1,
                'G' : 2,
                'T' : 3
        }
    
    def __init__(self, size=100):
        self.table = [[] for bucket in range(size)]
        self.size = size
        self.collisions = 0
        self.filled_slots = 0
        
    
        
    
    def insert(self, key, value):
        key_index = self.hash_kmer(key) % self.size
        bucket = self.table[key_index]
        
        if bucket:
            self.collisions += 1
        else:
            self.filled_slots += 1
            
        bucket = list(filter(lambda t: t[0] != key, bucket))
        bucket.append((key, value))
        

    def get(self, key):
        key_index = self.hash_kmer(key) % self.size
        bucket = self.table[key_index]
        
        for k,v in bucket:
            if k == key:
                return v
            
        return None
        
    def hash_kmer(self, kmer):
        k = len(kmer)
        hash_value = 0
        
        for i in range(0, k):
            hash_value += pow(4, k - i - 1) * self.hash_value(kmer[i]) 
        
        return hash_value    
        
        
        