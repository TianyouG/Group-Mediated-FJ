#pragma once

#include "fj/solver/linear_solver.hpp"

namespace fj {

class CgSolver : public LinearSolver {
 public:
  // Construct a CG solver with iteration and tolerance limits.
  CgSolver(Index max_iters, double tol);

  // Run (preconditioned) CG and return solver statistics.
  SolverStats Solve(const LinearOperator& A, const Vector& b, Vector& x,
                    const Preconditioner* M) const override;

  // Update max iterations.
  void set_max_iters(Index max_iters) { max_iters_ = max_iters; }
  // Update tolerance.
  void set_tolerance(double tol) { tol_ = tol; }

 private:
  Index max_iters_;
  double tol_;
};

}  // namespace fj
