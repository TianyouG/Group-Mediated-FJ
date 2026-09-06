#pragma once

#include <vector>

#include "fj/common/types.hpp"
#include "fj/graph/bipartite_csr.hpp"
#include "fj/graph/csr_graph.hpp"
#include "fj/solver/solver_stats.hpp"

namespace fj {

struct BliOptions {
  Index max_rounds = 200;
  Index max_updates = 0;
  double tolerance = 1e-8;
  double omega = 1.0;
  bool record_history = false;
};

struct BliResult {
  Vector x_u;
  Vector x_g;
  Index rounds = 0;
  Index updates = 0;
  double full_relative_residual = 0.0;
  bool converged = false;
  std::vector<ResidualTracePoint> residual_history;
};

class BliSolver {
 public:
  // Solve the weighted two-layer FJ system with local residual pushes.
  static BliResult Solve(const WeightedCsrGraph& user_graph,
                         const WeightedCsrGraph& group_graph,
                         const BipartiteCsr& bipartite,
                         const Vector& lambda_u, const Vector& lambda_g,
                         const Vector& b_u, const Vector& b_g,
                         const BliOptions& options);
};

}  // namespace fj
