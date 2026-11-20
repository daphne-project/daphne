g++ bfs.cpp -o bfs_omp `pkg-config --cflags eigen3` -O3 -fopenmp

#g++ bfs.cpp -o bfs_seq `pkg-config --cflags eigen3` -O3 -DEIGEN_DONT_PARALLELIZE 
