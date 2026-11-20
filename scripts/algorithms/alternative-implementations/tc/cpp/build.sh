g++ triangle_count.cpp -o tc_omp `pkg-config --cflags eigen3` -O3 -fopenmp

g++ triangle_count.cpp -o tc_seq `pkg-config --cflags eigen3` -O3 -DEIGEN_DONT_PARALLELIZE 
