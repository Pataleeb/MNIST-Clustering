# MNIST-Clustering
Multi-class classifications on the MNIST dataset: http://yann.lecun.com/exdb/mnist/

The MNIST database of handwritten digits has a training set of 60,000 examples and a test set of 10,000 examples. We use the clasters k=10 and standardize the features by dividing the values of the features by 255.

Purity score = corrected assigned samples/ size of the cluster 

1. Reporting the purity score for each cluster using the Euclidean distance as a metric for clustering.

2. Thresholding: threshold the pxels to convert the image into binary-valued pixel. If the pixel values are above 128, they are assigned as "1" and otherwise as "0". Now the distance matrix becomes Hamming distance and purity score for each cluster.



