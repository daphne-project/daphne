import sys
import time
from scipy.io import mmread
import numpy as np

filename = sys.argv[1]
G = mmread(filename)
start = time.time()
G_square = G @ G
nb_triangles = G_square.multiply(G).sum() / 3.0
fin = time.time()
print(fin - start)
