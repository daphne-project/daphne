#include <iostream>
#include <Eigen/SparseCore>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <unsupported/Eigen/SparseExtra>
#include <chrono>


typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMatR;

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cout << "Usage: bin mat.mtx size" << std::endl;
    return 1;
  }
  std::string filename = argv[1];

  int n = atoi(argv[2]);
  int maxi = 250;

  double alpha = 0.85;
  double one_minus_alpha = 1 - alpha;

  SpMatR G(n, n); 
  if (!loadMarket(G, filename))
    std::cout << "could  not load mtx file" << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  Eigen::VectorXd p(n);
  p = Eigen::VectorXd::Ones(n);

  for (int i = 0; i < maxi; i++) {
    p = (G * p) * alpha + p * one_minus_alpha;
    p = p / p.sum();
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration<float>(stop - start);
  std::cout << duration.count() << std::endl;
  return 0;
}
