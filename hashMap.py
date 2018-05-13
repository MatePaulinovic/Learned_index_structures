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
    
    def __init__(self, size=100, kmer=True):
        self.table = [[] for bucket in range(int(size))]
        self.size = int(size)
        self.collisions = 0
        self.filled_slots = 0
        if kmer:
            self.hash_fun = self.hash_kmer
        else:
            self.hash_fun = self.hash_encoded
        
    
        
    
    def insert(self, key, value):
        key_index = self.hash_fun(key) % self.size
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
        
    def hash_kmer(self, kmer):
        k = len(kmer)
        hash_value = 0
        
        for i in range(0, k):
            hash_value += pow(4, k - i - 1) * self.hash_value(kmer[i]) 
        
        return hash_value    
        
    def hash_encoded(self, encoded):
        k = len(encoded)
        hash_value = 0
        
        for i in range(0, k):
            hash_value += pow(4, k - i - 1) * int(encoded[i])
        
        return hash_value    
        
        