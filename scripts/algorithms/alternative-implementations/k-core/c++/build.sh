set -xe

g++ k-core.cpp -o k_core_omp `pkg-config --cflags eigen3` -O3 -fopenmp

g++ k-core.cpp -o k_core_seq `pkg-config --cflags eigen3` -O3 -DEIGEN_DONT_PARALLELIZE 
