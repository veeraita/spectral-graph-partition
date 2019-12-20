# Spectral graph partition

This program implements a graph partition based on a spectral algorithm by Shi and Malik (2000).

Example graphs (obtained from the [Stanford Network Analysis Project (SNAP)](http://snap.stanford.edu/data/index.html)) are in the directory `graphs_processed`. For simplicity and for avoiding trivial solutions, 
each graph is processed so that it contains only the largest connected component of the original graph.

Instructions for running the algorithm from the command line:

- Install the necessary requirements: "pip install -r requirements.txt" (or use Anaconda)
- Navigate to the directory "code"
- Run the code with the command "python spectral_partition.py <graphID> <k>"
  (for example "python spectral_partition.py ca-QrGc 2")


**References**

Shi, J. and Malik, J. (2000). Normalized cuts and image segmentation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 22(8), 888-905.
