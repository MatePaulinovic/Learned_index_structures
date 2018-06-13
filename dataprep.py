# -*- coding: utf-8 -*-
"""
@author: matep
"""

import fastaParser as P
import sys

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
        
        ####
        counter = 1
        ##### 

        for r in records:
            k = 16
            offset = 0
            l = len(r.seq)
            
            kmer = r.seq[0:k]
            encoded_kmer = self.encode_nucleotides(kmer)
            hash_val = self.hash_nucleotides(encoded_kmer)
            index = k
            
            destName = destination + r.id + ".txt"
            f = open(destName, "a+")
            while index < l:
               f.write("{kmer},{value}\n".format(kmer=encoded_kmer, value=hash_val))
               encoded_kmer = self.encode_nucleotides_fast(encoded_kmer, r.seq, index)
               hash_val = self.hash_nucleotides_fast(hash_val, r.seq, index, k)
               index += 1
            f.close()
                
            print("Completed file" + str(counter) + "/" + str(len(records)))
            counter += 1
            
        ###REST
     
        
    def generate_kmers(self, k, sequence):
        if k <= 0:
            return []
        
        kmers = []
        for i in range(0, len(sequence) - k + 1):
            kmer = sequence[i:i + k]
            if 'N' in kmer:
                continue
            
            kmers.append(sequence[i:i + k])
        
        return kmers
        
    
    def encode_nucleotides_fast(self, kmer, sequence, index):
        return kmer[1:] + str(self.hash_value(sequence[index]))

    
    def encode_nucleotides(self, kmer):
        result = ""
        for n in kmer:
            result += str(self.hash_value(n))
        
        return result

    def hash_nucleotides_fast(self, prev_hash, sequence, index, k):
        tmp = prev_hash - pow(4, k - 1) * self.hash_value(sequence[index - k])
        tmp *= 4
        tmp += self.hash_value(sequence[index])
        
        return tmp
    
    def hash_nucleotides(self, seq):
        k = len(seq)
        hash_value = 0
        for i in range(0, k):
            hash_value += pow(4, k - i - 1) * int(seq[i])
            
        return hash_value
        
    def hash_kmer(self, kmer):
        k = len(kmer)
        hash_value = 0
        for i in range(0, k):
            hash_value += pow(4, k - i - 1) * self.hash_value(kmer[i]) 
        
        return hash_value     
    

    def hash_value(self, nucleotide):
        return self.hash_values.get(nucleotide, -1)


if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print("Passed wrong number of arguemnts")
        exit(1)
    
    dp = DataPreper()
    dp.prepare_file(sys.argv[1], sys.argv[2])
    #dp.prepare_file('./data/GRCh37.fna', './data/training_set/GRCh37/')

