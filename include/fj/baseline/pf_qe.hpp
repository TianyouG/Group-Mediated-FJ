#pragma once

#include <cstdint>

#include "fj/common/types.hpp"
#include "fj/graph/bipartite_csr.hpp"
#include "fj/graph/csr_graph.hpp"

namespace fj {

struct PfQeOptions {
  Index sample_users = 10000;
  Index sample_edges = 10000;
  Index forest_samples = 1000;
  Index max_walk_steps = 1000000;
  std::uint64_t seed = 1;
};

struct PfQeResult {
  double disagreement = 0.0;
  double internal_conflict = 0.0;
  double polarization = 0.0;
  double controversy = 0.0;
  Index sampled_users = 0;
  Index sampled_edges = 0;
  Index forest_samples = 0;
  Index walk_steps = 0;
};

class PfQeEstimator {
 public:
  // Estimate user-side FJ quantities with weighted partial rooted forests.
  static PfQeResult Estimate(const WeightedCsrGraph& user_graph,
                             const WeightedCsrGraph& group_graph,
                             const BipartiteCsr& bipartite,
                             const Vector& lambda_u, const Vector& lambda_g,
                             const Vector& s_u, const Vector& s_g,
                             const PfQeOptions& options);
};

}  // namespace fj
