using MatrixMarket
using SparseArrays
using SparseMatricesCSR

function G_broadcast_mult_c(G, c)
  cols = colvals(G)
  vals = nonzeros(G)
  m, n = size(G)
  maxs = zeros(n)
  for j = 1:m
     for i in nzrange(G, j)
        col = cols[i]
        val = vals[i]
        if val * c[j] > maxs[col]
          maxs[col] = val*c[j]
        end
     end
  end
  maxs
end

function cc(filename, maxi)
  G = MatrixMarket.mmread(filename, :csr)
  start = time_ns()
  c = vec(collect(1.0:1.0:float(size(G, 1))))

  for iter in 1:maxi
    x = G_broadcast_mult_c(G, c)
    c = max.(c, x)
  end
  fin = time_ns()
  println((fin - start) * 1e-9)
end

@assert(length(ARGS) == 1)
filename = ARGS[1]
maxi = 100
cc(filename, maxi)

