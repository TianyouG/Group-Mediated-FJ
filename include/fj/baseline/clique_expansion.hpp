#pragma once

#include "fj/graph/bipartite_csr.hpp"
#include "fj/graph/csr_graph.hpp"

namespace fj {

class CliqueExpansion {
 public:
  // Build a user-only graph by expanding each group into a clique.
  static WeightedCsrGraph Build(const BipartiteCsr& bipartite,
                                bool use_weights);
};

}  // namespace fj
