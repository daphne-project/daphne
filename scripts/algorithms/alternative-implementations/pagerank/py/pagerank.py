from scipy.io import mmread
from scipy.sparse import coo_array
import time
import sys
import numpy as np

def pagerank(filename, maxi=250):
    G = mmread(filename)
    start = time.time()
    n = G.shape[0]
    p = np.ones(n)
    alpha = 0.85
    one_minus_alpha = 1 - alpha

    for iter in range(maxi):
        p = G.dot(p) * alpha + p * one_minus_alpha 
        p = p / p.sum()
    end = time.time()
    print(end - start)

if __name__ == "__main__":
    args = sys.argv
    assert len(args) == 2
    filename = args[1]
    pagerank(filename)
