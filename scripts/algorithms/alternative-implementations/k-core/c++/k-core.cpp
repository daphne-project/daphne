#include <iostream>
#include <Eigen/SparseCore>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <unsupported/Eigen/SparseExtra>


typedef Eigen::SparseMatrix<int, Eigen::RowMajor> SpMatR;

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
  std::cout << "Finished reading" << std::endl;

  Eigen::VectorXi c(n);
  Eigen::VectorXi x(n);
  Eigen::VectorXi prev(n);

  c = Eigen::VectorXi::Ones(n);
  x = Eigen::VectorXi::Zero(n);
  int diff = 1;
  int k = 9;

  while (diff != 0) {
    prev = c;
    x = G * c;
    c = (x.array() >= k).cast<int>().matrix();
    diff = (c.array() != prev.array()).count();
  }
  // std::cout << c << std::endl;
  return 0;
}
