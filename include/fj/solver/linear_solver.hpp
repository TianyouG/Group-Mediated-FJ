#pragma once

#include "fj/common/types.hpp"
#include "fj/operators/linear_operator.hpp"
#include "fj/preconditioner/preconditioner.hpp"
#include "fj/solver/solver_stats.hpp"

namespace fj {

class LinearSolver {
 public:
  virtual ~LinearSolver() = default;

  // Solve Ax=b with an optional preconditioner, returning stats.
  virtual SolverStats Solve(const LinearOperator& A, const Vector& b, Vector& x,
                            const Preconditioner* M) const = 0;
};

}  // namespace fj
