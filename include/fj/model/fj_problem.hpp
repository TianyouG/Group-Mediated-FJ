#pragma once

#include "fj/common/types.hpp"
#include "fj/graph/bipartite_csr.hpp"
#include "fj/graph/csr_graph.hpp"

namespace fj {

struct FjProblem {
  Index n_users = 0;
  Index n_groups = 0;

  Vector s_u;
  Vector s_g;
  Vector lambda_u;
  Vector lambda_g;

  WeightedCsrGraph user_graph;
  WeightedCsrGraph group_graph;
  BipartiteCsr bipartite;

  // Compute b_u = lambda_u .* s_u.
  Vector b_u() const;
  // Compute b_g = lambda_g .* s_g.
  Vector b_g() const;
};

}  // namespace fj
