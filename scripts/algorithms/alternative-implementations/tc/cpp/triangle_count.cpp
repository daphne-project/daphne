#include <iostream>
#include <Eigen/SparseCore>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <unsupported/Eigen/SparseExtra>
#include <chrono>

typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SpMatC;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMatR;

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cout << "Usage: bin mat.mtx size" << std::endl;
    return 1;
  }
  std::string filename = argv[1];

  int n = atoi(argv[2]);

  SpMatR G(n, n); 
  if (!loadMarket(G, filename)) {
    std::cout << "could  not load mtx file" << std::endl;
    return 1;
  }

  auto start = std::chrono::high_resolution_clock::now();

  SpMatR G_square = G * G;
  double nb_triangles = G_square.cwiseProduct(G).sum() / 3.0;

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration<float>(stop - start);
  // std::cout << duration.count() << ", " << nb_triangles << std::endl;
  std::cout << duration.count() << std::endl;
  return 0;
}
