#pragma once

#include "fj/common/types.hpp"
#include "fj/operators/linear_operator.hpp"

namespace fj {

class DirectSolver {
 public:
  // Solve Ax=b using dense LDLT (small problems only).
  static Vector Solve(const LinearOperator& A, const Vector& b);
};

}  // namespace fj
