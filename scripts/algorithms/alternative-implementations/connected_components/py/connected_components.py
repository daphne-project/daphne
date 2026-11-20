import sys
import time
from scipy.io import mmread
from scipy.sparse import csr_matrix, csr_array
import numpy as np

def cc(filename, maxi=100):
    G = csr_matrix(mmread(filename))
    n = G.shape[1]
    start = time.time()
    c = np.array([list(map(lambda i: float(i), range(1, n + 1, 1)))])

    for iter in range(maxi):
        x = G.multiply(c.transpose()).max(axis=0)
        c = np.maximum(c, x.todense())
    end = time.time()
    print(end - start)
    #print(c.sum())

if __name__ == "__main__":
    args = sys.argv
    assert len(args) == 2
    filename = args[1]
    cc(filename)
