#include <iostream>
#include <Eigen/SparseCore>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <unsupported/Eigen/SparseExtra>
#include <chrono>

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMatR;
typedef SpMatR::InnerIterator InIterMatR;

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cout << "Usage: bin mat.mtx size" << std::endl;
    return 1;
  }
  std::string filename = argv[1];

  int n = atoi(argv[2]);

  SpMatR G(n, n); 
  if (!loadMarket(G, filename))
    std::cout << "could  not load mtx file" << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  Eigen::VectorXd c(n);
  for (int i = 0; i < n; i++) {
    c(i) = (double)(i + 1);
  }

  for (int iter = 0; iter < 100; iter++) {
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
    for (int row_id = 0; row_id < G.outerSize(); row_id++) {
      double tmp = c.coeffRef(row_id);
      for (InIterMatR i_(G, row_id); i_; ++i_) {
        if (tmp > x.coeffRef(i_.col())) {
          x.coeffRef(i_.col()) = tmp;
        }
      }
    }
    c = c.cwiseMax(x);
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration<float>(stop - start);
  std::cout << duration.count() << std::endl;
  return 0;
}
