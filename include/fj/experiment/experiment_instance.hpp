#pragma once

#include "fj/common/types.hpp"
#include "fj/graph/bipartite_csr.hpp"
#include "fj/graph/csr_graph.hpp"

namespace fj {

struct ExperimentInstance {
  WeightedCsrGraph user_graph;
  WeightedCsrGraph group_graph;
  BipartiteCsr bipartite;
  Vector group_sizes;
  Vector user_degrees;

  Vector s_u;
  Vector s_g;
  Vector lambda_u;
  Vector lambda_g;
  Vector b_u;
  Vector b_g;
};

}  // namespace fj
