# Learned index structures

One Paragraph of project description goes here


## Prerequisites

Python 3.5 + 

Required python packages:
* [BioPython](https://biopython.org/)
* [PyTorch](https://pytorch.org)

Both can be installed independently or through various package management systems (pip, pipy, conda).


## Installing

Clone the source locally:
```bash
git clone https://github.com/MatePaulinovic/Learned_index_structures/
cd Learned_index_structures
```
And the code is ready to be used.

All further code excepts the current directory position to be `./Learned_index_structures`

## Example usage

Download a sample fasta/fastq file. For example the [human genome](https://www.ncbi.nlm.nih.gov/genome/guide/human/). 
```
wget ftp://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/annotation/GRCh38_latest/refseq_identifiers/GRCh38_latest_genomic.fna.gz
```

Now unpack the dowloaded file to get the fasta file
```
gunzip GRCh38_latest_genomic.fna.gz
```

Parse the fasta/fastq file and construct a training dataset
```
python dataprep.py <source_file> <destination_file>
```

Train a RMI structure from the constructed dataset with the given hash size and save the trained model to a folder
```
python hybridTraining.py <dataset_file> <hash_map_size> <serialization_folder>
```

Test the RMI structure on the provided test dataset by inserting into a hash table with size length(test_dataset) * factor
```
python RMIStat.py <test_dataset> <factor>
```

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/MatePaulinovic/Learned_index_structures/blob/master/LICENSE) file for details

## Acknowledgments

Thanks to prof. Mile Šikić for his guidance during the development of this project.
