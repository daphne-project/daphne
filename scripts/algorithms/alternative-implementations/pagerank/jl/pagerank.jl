using MatrixMarket

function pagerank(filename, maxi)
  G = MatrixMarket.mmread(filename)
  n = size(G, 1)
  p = ones(n)
  alpha = 0.85
  one_minus_alpha = 1 - alpha

  start = time_ns()
  for iter in 1:maxi
    p = (G * p) * alpha + p * one_minus_alpha
    p = p / sum(p)
  end
  fin = time_ns()
  println((fin - start) * 1e-9)
end

@assert(length(ARGS) == 1)
filename = ARGS[1]
maxi = 250
pagerank(filename, maxi)

