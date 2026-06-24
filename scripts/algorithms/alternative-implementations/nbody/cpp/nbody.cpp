#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <chrono>

Eigen::MatrixXd calculate_acceleration_matrix(Eigen::MatrixXd position, Eigen::VectorXd mass, double gravity, double softening, int n) {

  Eigen::VectorXd ones = Eigen::VectorXd::Ones(n);
  Eigen::VectorXd x = position.col(0);
  Eigen::VectorXd y = position.col(1);

  Eigen::MatrixXd dx = (ones * x.transpose()).colwise() - x;
  Eigen::MatrixXd dy = (ones * y.transpose()).colwise() - y;

  Eigen::MatrixXd inv_r3(n, n);
  inv_r3 = (dx.array().square() + dy.array().square()).array() + softening * softening;
  inv_r3 = inv_r3.array().pow(-1.5);

  Eigen::MatrixXd acceleration(n, 2);
  acceleration.col(0) = gravity * (dx * inv_r3) * mass;
  acceleration.col(1) = gravity * (dy * inv_r3) * mass;

  return acceleration;
}


int main(int argc, char** argv) {
  int n = 1000;
  double gravity = 0.00001;
  double step_size = 20.0 / 1000.0;
  double half_step_size = 0.5 * step_size;
  double softening = 0.1;

  auto start = std::chrono::high_resolution_clock::now();

  auto mat = Eigen::MatrixXd::Random(n, 2);
  auto mat_ones = Eigen::MatrixXd::Ones(n, 2);

  Eigen::MatrixXd position(n, 2);
  Eigen::MatrixXd velocity(n, 2);
  Eigen::MatrixXd acceleration(n, 2);
  Eigen::VectorXd mass(n);

  position = 5.0 * (mat.array() - 5.0); 
  velocity = Eigen::MatrixXd::Zero(n, 2);
  acceleration = Eigen::MatrixXd::Zero(n, 2);
  mass = 500.0 * Eigen::VectorXd::Ones(n);

  position.row(0) = Eigen::MatrixXd::Zero(1, 2);
  velocity.row(0) = Eigen::MatrixXd::Zero(1, 2);
  mass(0) = 10000.0;


  double sum_mass = mass.array().sum();
  auto com_p = (position.colwise() + mass).colwise().sum() / sum_mass;
  auto com_v = (velocity.colwise() + mass).colwise().sum() / sum_mass;

  position.rowwise() -= com_p;
  velocity.rowwise() -= com_v;

  for (int iter = 0; iter < 400; iter++) {
    // std::cout << iter << std::endl;

    velocity = velocity + acceleration * half_step_size;
    position = position + velocity * step_size;

    acceleration = calculate_acceleration_matrix(position, mass, gravity, softening, n);

    velocity = velocity + acceleration * half_step_size;
  }

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration<float>(stop - start);
  std::cout << duration.count() << std::endl;

  return 0;
}
