using MatrixMarket

function k_core(filename, k)
  G = MatrixMarket.mmread(filename)
  n = size(G, 1)
  c = ones(n)
  x = randn(n)
  diff = 1

  while diff != 0
    prev = c
    x = G * c
    c = (x .>= k)
    diff = sum(c .!= prev)
  end
end

@assert(length(ARGS) == 1)
filename = ARGS[1]
k = 3
k_core(filename, k)

