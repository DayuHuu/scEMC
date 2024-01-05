# scEMC

This is the source code for **scEMC: Effective multi-modal clustering method via skip aggregation network for parallel scRNA-seq and scATAC-seq data**. We propose an effective multi-modal clustering model scEMC for parallel scRNA and scATAC data. Concretely, we have devised a skip aggregation network (SAN) to simultaneously learn global structural information among cells and integrate data from diverse modalities. This manuscript is currently under peer review for publication in the journal "Briefings in Bioinformatics". If you have any questions about the code, please send an email to hzauhdy@gmail.com

# Requirements

Python --- 3.7.0

Numpy --- 1.21.6

Torch --- 1.13.1 

Scikit-learn --- 1.0.2

Scipy --- 1.7.3

Scanpy --- 1.9.3

# Guidance for running code
## The Source of datasets 

An example dataset, named 'BMNC.mat', has been provided. Furthermore, the sources for the other datasets are provided within our manuscript:

## Examples

The example expression matrix data.tsv of dataset Biase is put into data/Biase. To change datasets, you should type the iuput of code:
```python
parser.add_argument('--dataset_str', default='Biase', type=str, help='name of dataset')

parser.add_argument('--n_clusters', default=3, type=int, help='expected number of clusters')

parser.add_argument('--label_path', default='data/Biase/label.ann', type=str, help='true labels')

# ... other arguments ...
```
## Run 
```python
python scEMC.py
```


