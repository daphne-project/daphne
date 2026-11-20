
function calculate_acceleration_matrix(P::Matrix{Float64}, M::Matrix{Float64}, gravity::Float64, softening::Float64, n::Integer)
  x = P[:,1]
  y = P[:,2]
  dx = transpose(x) .- x 
  dy = transpose(y) .- y 
  inv_r3 = (dx.^2 + dy.^2 .+ softening^2).^(-1.5)

	ax = gravity .* (dx * inv_r3) * M
	ay = gravity .* (dy * inv_r3) * M

  return [ax ay]
end

function main()

  n = 1000
  gravity = 0.00001
  step_size = 20.0 / 1000.0
  half_step_size = 0.5 * step_size
  softening = 0.1

  start = time_ns()

  position = 5.0 .* (randn(n, 2) .- 5.0)
  velocity = zeros(n, 2)
  acceleration = zeros(n, 2)
  mass = 500.0 .* ones(n, 1)

  position[1, 1] = 0.0
  position[1, 2] = 0.0
  velocity[1, 1] = 0.0
  velocity[1, 2] = 0.0
  mass[1] = 10000.0

  com_p = sum(mass .* position, dims=1) / sum(mass)
  com_v = sum(mass .* velocity, dims=1) / sum(mass)

  position = position .- com_p
  velocity = velocity .- com_v

  for i in 1:400
    velocity = velocity .+ acceleration .* half_step_size
    position = position .+ velocity .* step_size

    acceleration = calculate_acceleration_matrix(position, mass, gravity, softening, n)

    velocity = velocity .+ acceleration .* half_step_size
  end
  fin = time_ns()
  println((fin - start) * 1e-9)
end

main()
