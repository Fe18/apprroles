# Structural Graph Representations based on Multiscale Local Network Topologies

### Information
This code provides an easy-to-follow implementation for calculating node descriptors, resp. graph representations, as described in the paper "Structural Graph Representations based on Multiscale Local Network Topologies" that has been published at the 2019 Web Intelligence Conference. However, we simplified the code such that everything can be computed within a single python environment. The original results reported in the paper have been achieved by using the APPR procedure from the Ligra framework by Julian Shun. Note that this framework is based on C and hence is expected to achieve better performance in terms of computation time. Also, we omitted the parameter search which has originally been done with a stratified kfold cross validation.

### Requirements
The following packages are required (the given versions should work):  
matplotlib==3.1.0  
networkx==2.3  
scikit-learn==0.21.2  
scipy==1.3.0  
numpy==1.16.4  

### Parameters
The apprroles.py script is the starter and supports two modes, i.e., graph-classification and role-descriptors. Therefore ```graph-classification``` , resp. ```role-descriptors``` must be given as positional arguments. For both modes, you can pass the following arguments:
* ```--dataset```: The graph dataset as adjacency list (see e.g. datasets/MUTAG/* for formatting and naming examples)
* ```--delimiter```: The delimiter for the adjacency list file
* ```--alphas```: A list of alpha parameters that shall be used for the APPR computation
* ```--epsilon```: The approximation error parameter for the APPR computation

Additionally, for the graph-classification mode you may also pass
* ```--k```: The dimensionality of the resulting graph representations
* ```--test-fraction```: The fraction of graphs that shall be used for testing

### Usage Examples
E.g. for graph classification on the nci1 dataset you may use  
```python apprroles.py graph-classification --dataset nci1 --delimiter , --alphas 0.1 0.5 0.9 --k 15```,

and for calculating and plotting 1-dimensional role descriptors you may use  
```python apprroles.py role-descriptors --dataset barbell```