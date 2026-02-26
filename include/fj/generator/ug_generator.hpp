#pragma once

#include <cstdint>

#include "fj/common/types.hpp"
#include "fj/graph/bipartite_csr.hpp"

namespace fj {

struct UgConfig {
  // Number of users.
  Index n_users = 0;
  // Number of groups.
  Index n_groups = 0;

  // Power-law exponent for group size distribution.
  double alpha = 2.0;
  // Minimum group size.
  Index s_min = 1;
  // Maximum group size.
  Index s_max = 1;

  // Mean user participation (Poisson).
  double user_mean = 1.0;
  // Maximum user participation.
  Index user_r_max = 1;

  // RNG seed.
  uint64_t seed = 1;
};

struct UgResult {
  // User-group bipartite graph.
  BipartiteCsr bipartite;
  // Group sizes (membership counts).
  Vector group_sizes;
  // User degrees (participation counts).
  Vector user_degrees;
};

class UgGenerator {
 public:
  // Generate a bipartite user-group graph.
  static UgResult Generate(const UgConfig& cfg);
};

}  // namespace fj
