#include "fj/baseline/direct_solver.hpp"

#include <stdexcept>

#include "fj/common/types.hpp"

namespace fj {

Vector DirectSolver::Solve(const LinearOperator& A, const Vector& b) {
  // Form a dense matrix via operator application and factorize with LDLT.
  if (A.rows() != A.cols()) {
    throw std::invalid_argument("DirectSolver requires square operator");
  }
  if (b.size() != A.rows()) {
    throw std::invalid_argument("DirectSolver size mismatch");
  }

  const Index n = A.rows();
  Eigen::MatrixXd dense(n, n);
  Vector e = Vector::Zero(n);
  Vector col(n);

  for (Index i = 0; i < n; ++i) {
    if (i > 0) {
      e[i - 1] = 0.0;
    }
    e[i] = 1.0;
    A.Apply(e, col);
    dense.col(i) = col;
  }

  Eigen::LDLT<Eigen::MatrixXd> ldlt(dense);
  if (ldlt.info() != Eigen::Success) {
    throw std::runtime_error("DirectSolver LDLT factorization failed");
  }
  Vector x = ldlt.solve(b);
  if (ldlt.info() != Eigen::Success) {
    throw std::runtime_error("DirectSolver solve failed");
  }
  return x;
}

}  // namespace fj
