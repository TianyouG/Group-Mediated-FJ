#pragma once

#include "fj/common/types.hpp"

namespace fj {

// Statistics returned by linear solvers.
struct SolverStats {
  // Iteration count used by the solver.
  Index iterations = 0;
  // Final residual norm.
  double residual_norm = 0.0;
  // Final relative residual norm.
  double relative_residual = 0.0;
  // Total wall-clock time (seconds).
  double seconds = 0.0;
  // Convergence flag.
  bool converged = false;
};

}  // namespace fj
