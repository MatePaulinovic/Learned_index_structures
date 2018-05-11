# -*- coding: utf-8 -*-
"""
@author: matep
"""
from Bio import SeqIO
import InvalidFormatError
import os.path

class FastaParser():
     
    def __init__(self):
        self.records = []
        self.numOfRecords = 0
    
    
    def get_records(self):
        """
        Returns parsed records of the specified file.
        If none were read returns an empty list.
        """
        
        return self.records
    
    
    def read_input(self, path):
        """
        Reads the specified fasta file and stores the read records in its 
        internal  variables. Supported fasta file extensions are: 
        (".fna", ".fasta", ".fas", ".fa", ".seq", ".fsa", ".ffn", ".frn")
        """
        #path = './data/GRCh37.fna'
        
        if not path.endswith((".fna", ".fasta", ".fas", ".fa", ".seq", ".fsa", ".ffn", ".frn")):
            raise InvalidFormatError("File is of unsupported type")
            
        try:
            counter = 0
            records = []
            with open(path, mode='r') as handle:
    
                for record in SeqIO.parse(handle, 'fasta'):
            
                    identifier = record.id
                    description = record.description
                    sequence = record.seq = record.seq.upper()
                    
                    ###############################
                    if 'N' in sequence:
                        continue
                    ###############################
                    counter += 1
                    print('-----------------------------------------------------------------')
                    print('Processing the record {}:'.format(identifier))
                    print('Its description is: \n{}'.format(description))
                    amount_of_nucleotides = len(sequence)
                    print('Its sequence contains {} nucleotides.'.format(amount_of_nucleotides))
            
                    records.append(record)
                    
                  
          
            self.records = records
            self.numOfRecords = len(records)
            return records
        
        except IOError as e:
            print("FAILED")
            pass
        
"""       
p = FastaParser()
p.read_input('./data/GrCh37.fna')
for r in p.records:
    print('A' in r.seq or 'C' in r.seq or 'T' in r.seq or 'G' in r.seq)
"""
