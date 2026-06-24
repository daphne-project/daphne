g++ connected_components.cpp -o cc_omp `pkg-config --cflags eigen3` -O3 -fopenmp

g++ connected_components.cpp -o cc_seq `pkg-config --cflags eigen3` -O3 -DEIGEN_DONT_PARALLELIZE 
