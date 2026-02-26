#include "fj/solver/inner_solver.hpp"

#include <stdexcept>

#include "fj/common/timer.hpp"

namespace fj {

// Initialize a diagonal inner solver.
DiagonalInnerSolver::DiagonalInnerSolver(const Vector& diag) : diag_(diag) {
  if (diag_.size() == 0) {
    throw std::invalid_argument("DiagonalInnerSolver diag cannot be empty");
  }
}

void DiagonalInnerSolver::Solve(const Vector& rhs, Vector& sol) const {
  // Solve diag * sol = rhs element-wise.
  if (rhs.size() != diag_.size()) {
    throw std::invalid_argument("DiagonalInnerSolver size mismatch");
  }
  Timer timer;
  sol.resize(rhs.size());
  for (Index i = 0; i < rhs.size(); ++i) {
    if (diag_[i] != 0.0) {
      sol[i] = rhs[i] / diag_[i];
    } else {
      sol[i] = rhs[i];
    }
  }
  stats_.solves += 1;
  stats_.seconds += timer.ElapsedSeconds();
}

// Reset accumulated stats for diagonal solves.
void DiagonalInnerSolver::ResetStats() const { stats_ = InnerSolveStats{}; }

// Return accumulated stats for diagonal solves.
InnerSolveStats DiagonalInnerSolver::Stats() const { return stats_; }

// Initialize an inner solver that delegates to a linear solver.
IterativeInnerSolver::IterativeInnerSolver(const LinearOperator& A,
                                           const LinearSolver& solver,
                                           const Preconditioner* M)
    : A_(&A), solver_(&solver), M_(M), size_(A.rows()) {
  if (A.rows() != A.cols()) {
    throw std::invalid_argument("Inner operator must be square");
  }
  if (M_ && M_->size() != A.rows()) {
    throw std::invalid_argument("Inner preconditioner size mismatch");
  }
}

void IterativeInnerSolver::Solve(const Vector& rhs, Vector& sol) const {
  // Solve using the configured iterative solver and update stats.
  if (rhs.size() != size_) {
    throw std::invalid_argument("IterativeInnerSolver size mismatch");
  }
  sol.setZero(size_);
  SolverStats stats = solver_->Solve(*A_, rhs, sol, M_);
  stats_.solves += 1;
  stats_.iterations += stats.iterations;
  stats_.seconds += stats.seconds;
}

// Reset accumulated stats for iterative solves.
void IterativeInnerSolver::ResetStats() const { stats_ = InnerSolveStats{}; }

// Return accumulated stats for iterative solves.
InnerSolveStats IterativeInnerSolver::Stats() const { return stats_; }

}  // namespace fj
