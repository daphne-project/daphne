g++ nbody.cpp -o nbody_omp `pkg-config --cflags eigen3` -O3 -fopenmp

g++ nbody.cpp -o nbody `pkg-config --cflags eigen3` -O3 -DEIGEN_DONT_PARALLELIZE 
