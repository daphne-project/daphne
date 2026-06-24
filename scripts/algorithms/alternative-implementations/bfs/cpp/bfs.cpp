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
  int maxi = 200;

  SpMatR G(n, n); 
  if (!loadMarket(G, filename)) {
    std::cout << "could  not load mtx file" << std::endl;
    return 1;
  }

  Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
  x(0) = 1.0;
  Eigen::VectorXd ones = Eigen::VectorXd::Ones(n);

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < maxi; i++) {
    x = ones.array().min((x + G * x).array());
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration<float>(stop - start);
  std::cout << duration.count() << std::endl;
  return 0;
}
