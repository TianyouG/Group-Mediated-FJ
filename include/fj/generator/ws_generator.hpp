#pragma once

#include <cstdint>

#include "fj/common/types.hpp"
#include "fj/graph/csr_graph.hpp"

namespace fj {

struct WsConfig {
  // Number of nodes.
  Index n = 0;
  // Ring lattice degree (must be even).
  Index k = 0;
  // Rewiring probability.
  double p = 0.0;
  // Edge weight.
  double weight = 1.0;
  // RNG seed.
  uint64_t seed = 1;
};

class WsGenerator {
 public:
  // Generate a WS graph using the given configuration.
  static WeightedCsrGraph Generate(const WsConfig& cfg);
};

}  // namespace fj
