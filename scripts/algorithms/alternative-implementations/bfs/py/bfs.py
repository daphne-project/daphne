import sys
import time
import numpy as np
from scipy.io import mmread

def bfs(filename, maxi=2000):
    G = mmread(filename)
    n = G.shape[0]
    x = np.zeros(n)
    x[0] = 1.0
    one = np.ones(n)
    

    start = time.time()
    for iter in range(maxi):
        x = np.minimum(1.0, x + G.dot(x))
    end = time.time()
    print(end - start)
    print(x.sum())

if __name__ == "__main__":
    args = sys.argv
    assert len(args) == 2
    filename = args[1]
    bfs(filename)
