using MatrixMarket
using SparseArrays

G = mmread(ARGS[1])
start = time_ns()
G_square = G * G
nb_triangles = sum(G_square .* G) / 3.0
fin = time_ns()
println((fin - start) * 1e-9)
