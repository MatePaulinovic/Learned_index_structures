# -*- coding: utf-8 -*-
"""
@author: matep
"""

import fastaParser as P

class DataPreper():
    
    def __init__(self):
        self.parser = P.FastaParser()
        self.hash_values = {
                'A' : 0,
                'C' : 1,
                'G' : 2,
                'T' : 3
        }
    

    def prepare_file(self, source, destination):
        try:
            records = self.parser.read_input(source)
        except IOException as e:
            #HANDLE ERROR
            return
        
        batch = 10000
        
        #print("Imam za obraditi {}  recrods".format(len(records)))
        
        for r in records:
            k = 16
            offset = 0

            while offset < len(r.seq):
        
                kmers = self.generate_kmers(k, r.seq[offset : offset + batch + k - 1])
                #print("Offset {}".format(offset))
                #print("num of kmers: {}".format(len(kmers)))
                kmer_hash = []
                for kmer in kmers:
                    kmer_hash.append((self.encode_nucleotides(kmer), self.hash_kmer(kmer)))
             
                destName = destination + r.id + ".txt"
                f = open(destName, "a+")
                for k_h in kmer_hash:
                    f.write("{kmer},{value}\n".format(kmer=k_h[0], value=k_h[1]))
                f.close()
           
                offset += batch
    
            
        ###REST
        return "NAPISAH"
        
    def generate_kmers(self, k, sequence):
        if k <= 0:
            return []
        
        kmers = []
        for i in range(0, len(sequence) - k + 1):
            kmers.append(sequence[i:i + k])
        
        return kmers
        
    
    def encode_nucleotides(self, kmer):
        result = ""
        for n in kmer:
            result += str(self.hash_value(n))
        
        return result

    
    def hash_kmer(self, kmer):
        k = len(kmer)
        hash_value = 0
        for i in range(0, k):
            hash_value += pow(4, k - i - 1) * self.hash_value(kmer[i]) 
        
        return hash_value     
    

    def hash_value(self, nucleotide):
        return self.hash_values.get(nucleotide, -1)


dp = DataPreper()
dp.prepare_file('./data/GrCh37.fna', './data/training_set/')
