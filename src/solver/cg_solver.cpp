#include "fj/solver/cg_solver.hpp"

#include <cmath>
#include <stdexcept>

#include "fj/common/timer.hpp"

namespace fj {

// Initialize CG solver parameters.
CgSolver::CgSolver(Index max_iters, double tol, bool record_history)
    : max_iters_(max_iters),
      tol_(tol),
      record_history_(record_history) {}

SolverStats CgSolver::Solve(const LinearOperator& A, const Vector& b, Vector& x,
                            const Preconditioner* M) const {
  // Solve Ax=b using (preconditioned) conjugate gradients.
  if (A.rows() != A.cols()) {
    throw std::invalid_argument("CG requires a square operator");
  }
  if (b.size() != A.rows()) {
    throw std::invalid_argument("CG size mismatch");
  }
  if (M && M->size() != A.rows()) {
    throw std::invalid_argument("Preconditioner size mismatch");
  }

  if (x.size() != A.rows()) {
    x.setZero(A.rows());
  }

  Timer timer;
  SolverStats stats;

  Vector r(A.rows());
  Vector Ap(A.rows());
  A.Apply(x, Ap);
  r = b - Ap;

  double b_norm = b.norm();
  if (b_norm == 0.0) {
    b_norm = 1.0;
  }

  double r_norm = r.norm();
  double rel_res = r_norm / b_norm;
  if (record_history_) {
    stats.residual_history.push_back({0, rel_res, timer.ElapsedSeconds()});
  }
  if (rel_res <= tol_) {
    stats.converged = true;
    stats.iterations = 0;
    stats.residual_norm = r_norm;
    stats.relative_residual = rel_res;
    stats.seconds = timer.ElapsedSeconds();
    return stats;
  }

  Vector p(A.rows());
  Vector z(A.rows());

  double rz_old = 0.0;
  double rr_old = 0.0;

  if (M) {
    M->Apply(r, z);
    p = z;
    rz_old = r.dot(z);
  } else {
    p = r;
    rr_old = r.squaredNorm();
  }

  Index performed_iterations = 0;
  for (Index iter = 0; iter < max_iters_; ++iter) {
    A.Apply(p, Ap);
    double denom = p.dot(Ap);
    if (denom == 0.0) {
      break;
    }
    double alpha = 0.0;
    if (M) {
      alpha = rz_old / denom;
    } else {
      alpha = rr_old / denom;
    }

    x += alpha * p;
    r -= alpha * Ap;
    performed_iterations = iter + 1;

    r_norm = r.norm();
    rel_res = r_norm / b_norm;
    if (record_history_) {
      stats.residual_history.push_back(
          {performed_iterations, rel_res, timer.ElapsedSeconds()});
    }

    if (rel_res <= tol_) {
      stats.converged = true;
      stats.iterations = performed_iterations;
      stats.residual_norm = r_norm;
      stats.relative_residual = rel_res;
      stats.seconds = timer.ElapsedSeconds();
      return stats;
    }

    if (M) {
      M->Apply(r, z);
      double rz_new = r.dot(z);
      double beta = rz_new / rz_old;
      p = z + beta * p;
      rz_old = rz_new;
    } else {
      double rr_new = r.squaredNorm();
      double beta = rr_new / rr_old;
      p = r + beta * p;
      rr_old = rr_new;
    }
  }

  stats.converged = false;
  stats.iterations = performed_iterations;
  stats.residual_norm = r_norm;
  stats.relative_residual = rel_res;
  stats.seconds = timer.ElapsedSeconds();
  return stats;
}

}  // namespace fj
