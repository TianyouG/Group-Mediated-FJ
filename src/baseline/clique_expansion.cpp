#include "fj/baseline/clique_expansion.hpp"

#include <algorithm>
#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fj {
namespace {

using EdgeKey = std::pair<Index, Index>;

struct EdgeKeyHash {
  size_t operator()(const EdgeKey& key) const noexcept {
    const size_t h1 = std::hash<Index>{}(key.first);
    const size_t h2 = std::hash<Index>{}(key.second);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
  }
};

EdgeKey CanonicalEdge(Index a, Index b) {
  return a < b ? EdgeKey{a, b} : EdgeKey{b, a};
}

}  // namespace

WeightedCsrGraph CliqueExpansion::Build(const BipartiteCsr& bipartite,
                                        bool use_weights) {
  // Build a user graph by expanding each group into a clique.
  const Index n_users = bipartite.num_users();
  const Index n_groups = bipartite.num_groups();
  if (n_users <= 0 || n_groups <= 0) {
    return WeightedCsrGraph(n_users);
  }

  const auto& row_ptr = bipartite.row_ptr();
  const auto& col_idx = bipartite.col_idx();
  const auto& values = bipartite.values();
  std::vector<std::vector<std::pair<Index, double>>> members(
      static_cast<size_t>(n_groups));

  // Gather group membership lists.
  for (Index u = 0; u < n_users; ++u) {
    const Index start = row_ptr[static_cast<size_t>(u)];
    const Index end = row_ptr[static_cast<size_t>(u + 1)];
    for (Index idx = start; idx < end; ++idx) {
      const Index g = col_idx[static_cast<size_t>(idx)];
      const double w = values[static_cast<size_t>(idx)];
      members[static_cast<size_t>(g)].emplace_back(u, w);
    }
  }

  std::unordered_map<EdgeKey, double, EdgeKeyHash> edge_weights;

  const Vector& group_degree = bipartite.group_degree();

  // Accumulate clique edges per group.
  for (Index g = 0; g < n_groups; ++g) {
    const auto& group = members[static_cast<size_t>(g)];
    if (group.size() < 2) {
      continue;
    }
    const double denom = use_weights ? group_degree[g] : 1.0;
    if (use_weights && denom <= 0.0) {
      continue;
    }
    for (size_t i = 0; i + 1 < group.size(); ++i) {
      const Index u = group[i].first;
      const double wu = use_weights ? group[i].second : 1.0;
      for (size_t j = i + 1; j < group.size(); ++j) {
        const Index v = group[j].first;
        const double wv = use_weights ? group[j].second : 1.0;
        const double w = use_weights ? (wu * wv / denom) : 1.0;
        const EdgeKey key = CanonicalEdge(u, v);
        edge_weights[key] += w;
      }
    }
  }

  std::vector<Triplet> triplets;
  triplets.reserve(edge_weights.size());
  // Convert aggregated edges to triplets.
  for (const auto& kv : edge_weights) {
    const Index u = kv.first.first;
    const Index v = kv.first.second;
    if (u == v) {
      continue;
    }
    triplets.emplace_back(u, v, kv.second);
  }

  // Build a symmetric weighted graph from the triplets.
  return WeightedCsrGraph::FromTriplets(n_users, triplets, true);
}

}  // namespace fj
