import numpy as np
import time

n = 1000
gravity = 0.00001
step_size = 20.0 / 1000.0
half_step_size = 0.5 * step_size
softening = 0.1

start = time.time()

position = 5.0 * (np.random.rand(n, 2) - 5.0)
velocity = np.zeros((n, 2)) #1 * (np.random.rand(n, 2) - 5)
acceleration = np.zeros((n, 2))
mass = 500.0 * np.ones((n, 1))

position[0] = np.array([0, 0])
velocity[0] = np.array([0, 0])
mass[0] = 10000.0

com_p = np.sum(np.multiply(mass, position), axis=0) / np.sum(mass, axis=0)
com_v = np.sum(np.multiply(mass, velocity), axis=0) / np.sum(mass, axis=0)

for p in position: p -= com_p
for v in velocity: v -= com_v


def calculate_acceleration_matrix(P, M, gravity, softening):
	x = P[:,0:1]
	y = P[:,1:2]
	dx = x.T - x
	dy = y.T - y
	inv_r3 = (dx**2 + dy**2 + softening**2) ** (-1.5)
	ax = gravity * (dx * inv_r3) @ M
	ay = gravity * (dy * inv_r3) @ M
	return np.hstack((ax, ay))

for i in range(400):
	velocity += acceleration * half_step_size
	position += velocity * step_size

	acceleration = calculate_acceleration_matrix(
			position, mass, gravity, softening)

	velocity += acceleration * half_step_size
	#print(i)

end = time.time()
print(end - start)

