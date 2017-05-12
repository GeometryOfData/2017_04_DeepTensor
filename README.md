# DeepTensor

Unsupervised learning of tensor product structures using deep networks.

Simple example: src/test_002.py, builds a representations for an image.

# Abstract

This project studies a representation of tensor structures, such as matrices and tables. We are interest in unsupervised learning of the structure of data (organization of rows and columns), and in applications to compression, denoising, and supervised learning.

The idea is loosely related to work on [Sampling, denoising and compression of matrices by coherent matrix organization](https://doi.org/10.1016/j.acha.2012.02.001) (a.k.a. "The Questionnaire").

The typical d-dimensional dataset would be composed of d-dimensional "coordinates" (e.g. row and column in the case of tables), and the entry in each coordinate (e.g. the value at this coordinate of the matrix, or an RBG vector in an image). Each coordinate would be an index in no particular order. For example, consider an image, where all the rows and columns have been permuted - that image can be though of as a collection of coordinates (i,j) and the color of the pixel at each coordinate (r(i,j),g(i,j),b(i,j)). Some of the pixel value may be missing. The algorithm would attempt to discover an embedding of the rows and an embedding of the columns which reflect their structure (which may be more or less related to the correct order of row and correct order of columns). 

# Network structure

The network takes d indices (the coordinates) and translates each index to a vector (the embedding of that index for that dimension). The values of the vectors are then combined (entry-wise sum of the vectors in the simple examples, and more elaborate tensor products in general). The combined vectors are fed into a deep network which predicts the value of at that coordinate. In the training process, both the network weights and the list of vectors corresponding to each index are updated (analogously to word2vec networks), as the networks is trained to improve the predictions. 
The idea is that the structural constraint would “force” the network to discover meaningful representations that reflect a structure in each of the d input spaces. Each index in each of the input spaces would have a vector associated with it, and these vectors would reveal the structure. In the unsupervised learning application, we are more interested in the structure that is revealed than in the quality of the predictions. 

