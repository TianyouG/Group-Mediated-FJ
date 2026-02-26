#include "fj/metrics/metrics.hpp"

#include <stdexcept>

namespace fj {

double RelativeResidual(const LinearOperator& A, const Vector& x,
                        const Vector& b) {
  // Compute ||b - A x|| / ||b|| with safety for ||b||=0.
  if (A.rows() != A.cols()) {
    throw std::invalid_argument("RelativeResidual expects square operator");
  }
  if (x.size() != A.rows() || b.size() != A.rows()) {
    throw std::invalid_argument("RelativeResidual size mismatch");
  }
  Vector Ax(A.rows());
  A.Apply(x, Ax);
  Vector r = b - Ax;
  double denom = b.norm();
  if (denom == 0.0) {
    denom = 1.0;
  }
  return r.norm() / denom;
}

double GraphDisagreement(const WeightedCsrGraph& graph, const Vector& x) {
  // Compute x^T L x using the graph Laplacian.
  if (x.size() != graph.num_nodes()) {
    throw std::invalid_argument("GraphDisagreement size mismatch");
  }
  Vector Lx(graph.num_nodes());
  graph.laplacian_matvec(x, Lx);
  return x.dot(Lx);
}

double InternalConflict(const Vector& x, const Vector& s) {
  // Compute ||x - s||^2 with size checking.
  if (x.size() != s.size()) {
    throw std::invalid_argument("InternalConflict size mismatch");
  }
  return (x - s).squaredNorm();
}

double Polarization(const Vector& x) {
  // Compute sum of squared deviations from the mean.
  if (x.size() == 0) {
    return 0.0;
  }
  const double mean = x.mean();
  return (x.array() - mean).square().sum();
}

double Controversy(const Vector& x) {
  // Compute squared norm of x.
  return x.squaredNorm();
}

double Variance(const Vector& x) {
  // Compute population variance of x.
  if (x.size() == 0) {
    return 0.0;
  }
  double mean = x.mean();
  double var = (x.array() - mean).square().mean();
  return var;
}

}  // namespace fj
