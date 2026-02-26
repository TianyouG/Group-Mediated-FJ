#pragma once

#include <cstdint>

#include "fj/common/types.hpp"
#include "fj/graph/bipartite_csr.hpp"
#include "fj/graph/csr_graph.hpp"

namespace fj {

enum class GroupGraphMode {
  kNone = 0,
  kErdosRenyi = 1,
  kOverlapKnn = 2,
  kOverlapThreshold = 3,
};

struct GroupGraphConfig {
  // Group graph generation mode.
  GroupGraphMode mode = GroupGraphMode::kNone;
  // Number of groups (optional if bipartite is provided).
  Index n_groups = 0;

  // ER probability (used in kErdosRenyi).
  double p = 0.0;
  // k for overlap-kNN (used in kOverlapKnn).
  Index k = 0;
  // Edge weight scale.
  double weight = 1.0;
  // Minimum overlap ratio overlap/min(|g|,|h|) to keep an edge.
  double overlap_ratio_threshold = 0.5;
  // Global scale target for average weighted degree.
  double target_mean_degree = 1.0;

  // RNG seed.
  uint64_t seed = 1;
};

class GroupGraphGenerator {
 public:
  // Generate a group graph using the requested mode.
  static WeightedCsrGraph Generate(const GroupGraphConfig& cfg,
                                   const BipartiteCsr* bipartite);
};

}  // namespace fj
