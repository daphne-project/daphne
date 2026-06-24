using MatrixMarket
using SparseArrays

function G_mult_c(G, c)
  rows = rowvals(G)
  vals = nonzeros(G)
  m, n = size(G)
  result = fill(0.0, n)
  @Threads.threads for j = 1:n
     for i in nzrange(G, j)
        row = rows[i]
        val = vals[i]
        result[row] += val * c[j]
     end
  end
  result
end

function main(filename)
  G = mmread(filename)
  n = size(G, 1)
  x = zeros(n)
  x[1] = 1.0

  maxi = 2000
  start = time_ns()
  for iter in 1:maxi
    #x = min.(1.0, x .+ G * x)
    x = min.(1.0, x .+ G_mult_c(G, x))
  end
  fin = time_ns()
  println((fin - start) * 1e-9)
end

main(ARGS[1])
