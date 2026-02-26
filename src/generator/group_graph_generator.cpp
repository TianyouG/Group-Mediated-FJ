#include "fj/generator/group_graph_generator.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fj {
namespace {

Index ResolveGroupCount(const GroupGraphConfig& cfg, const BipartiteCsr* bipartite) {
  // Resolve group count from config or bipartite graph.
  if (cfg.n_groups > 0) {
    if (bipartite && bipartite->num_groups() != cfg.n_groups) {
      throw std::invalid_argument("GroupGraphConfig n_groups mismatch with bipartite");
    }
    return cfg.n_groups;
  }
  if (bipartite) {
    return bipartite->num_groups();
  }
  throw std::invalid_argument("GroupGraphConfig n_groups must be positive");
}

// Generate an empty group graph.
WeightedCsrGraph GenerateNone(Index n_groups) { return WeightedCsrGraph(n_groups); }

template <typename Emit>
void ForEachErdosRenyiEdge(const GroupGraphConfig& cfg, Index n_groups,
                           std::mt19937_64& rng, Emit emit) {
  // Enumerate undirected edges for an Erdos-Renyi graph.
  const double log_q = std::log(1.0 - cfg.p);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  Index v = 1;
  Index w = -1;
  while (v < n_groups) {
    const double r = dist(rng);
    const double skip = std::floor(std::log(1.0 - r) / log_q);
    w = w + 1 + static_cast<Index>(skip);
    while (w >= v && v < n_groups) {
      w -= v;
      ++v;
    }
    if (v < n_groups) {
      emit(v, w);
    }
  }
}

WeightedCsrGraph GenerateErdosRenyi(const GroupGraphConfig& cfg, Index n_groups) {
  // Generate an undirected Erdos-Renyi group graph.
  if (cfg.p <= 0.0) {
    return WeightedCsrGraph(n_groups);
  }
  if (cfg.weight <= 0.0) {
    throw std::invalid_argument("GroupGraphConfig weight must be positive");
  }

  std::vector<Index> row_counts(static_cast<size_t>(n_groups), 0);
  if (cfg.p >= 1.0) {
    for (Index i = 0; i < n_groups; ++i) {
      row_counts[static_cast<size_t>(i)] = n_groups - 1;
    }
  } else {
    std::mt19937_64 rng(cfg.seed);
    ForEachErdosRenyiEdge(cfg, n_groups, rng, [&](Index a, Index b) {
      row_counts[static_cast<size_t>(a)] += 1;
      row_counts[static_cast<size_t>(b)] += 1;
    });
  }

  std::vector<Index> row_ptr(static_cast<size_t>(n_groups + 1), 0);
  for (Index i = 0; i < n_groups; ++i) {
    row_ptr[static_cast<size_t>(i + 1)] =
        row_ptr[static_cast<size_t>(i)] + row_counts[static_cast<size_t>(i)];
  }

  const Index nnz = row_ptr.back();
  std::vector<Index> col_idx(static_cast<size_t>(nnz));
  std::vector<Scalar> values(static_cast<size_t>(nnz));
  std::vector<Index> offsets = row_ptr;

  auto add_edge = [&](Index u, Index v) {
    const Index pos = offsets[static_cast<size_t>(u)]++;
    col_idx[static_cast<size_t>(pos)] = v;
    values[static_cast<size_t>(pos)] = cfg.weight;
  };

  if (cfg.p >= 1.0) {
    for (Index i = 0; i < n_groups; ++i) {
      for (Index j = 0; j < n_groups; ++j) {
        if (i == j) {
          continue;
        }
        add_edge(i, j);
      }
    }
  } else {
    std::mt19937_64 rng(cfg.seed);
    ForEachErdosRenyiEdge(cfg, n_groups, rng, [&](Index a, Index b) {
      add_edge(a, b);
      add_edge(b, a);
    });
  }

  return WeightedCsrGraph::FromCsr(n_groups, std::move(row_ptr),
                                   std::move(col_idx), std::move(values));
}

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

WeightedCsrGraph GenerateOverlapKnn(const GroupGraphConfig& cfg,
                                    const BipartiteCsr& bipartite,
                                    Index n_groups) {
  // Generate a group graph using top-k shared-membership overlaps.
  if (cfg.k <= 0) {
    return WeightedCsrGraph(n_groups);
  }
  if (cfg.weight <= 0.0) {
    throw std::invalid_argument("GroupGraphConfig weight must be positive");
  }

  const Index n_users = bipartite.num_users();
  const auto& bip_row_ptr = bipartite.row_ptr();
  const auto& bip_col_idx = bipartite.col_idx();

  std::vector<std::vector<Index>> user_groups(static_cast<size_t>(n_users));
  for (Index u = 0; u < n_users; ++u) {
    std::vector<Index>& groups = user_groups[static_cast<size_t>(u)];
    const Index start = bip_row_ptr[static_cast<size_t>(u)];
    const Index end = bip_row_ptr[static_cast<size_t>(u + 1)];
    for (Index idx = start; idx < end; ++idx) {
      groups.push_back(bip_col_idx[static_cast<size_t>(idx)]);
    }
  }

  std::vector<std::unordered_map<Index, double>> counts(
      static_cast<size_t>(n_groups));
  for (const auto& groups : user_groups) {
    if (groups.size() < 2) {
      continue;
    }
    for (size_t i = 0; i + 1 < groups.size(); ++i) {
      Index g1 = groups[i];
      for (size_t j = i + 1; j < groups.size(); ++j) {
        Index g2 = groups[j];
        if (g1 == g2) {
          continue;
        }
        counts[static_cast<size_t>(g1)][g2] += 1.0;
        counts[static_cast<size_t>(g2)][g1] += 1.0;
      }
    }
  }

  std::unordered_map<EdgeKey, double, EdgeKeyHash> edge_weights;
  edge_weights.reserve(static_cast<size_t>(n_groups) * 2);

  for (Index g = 0; g < n_groups; ++g) {
    const auto& cmap = counts[static_cast<size_t>(g)];
    if (cmap.empty()) {
      continue;
    }
    std::vector<std::pair<Index, double>> neighbors;
    neighbors.reserve(cmap.size());
    for (const auto& kv : cmap) {
      neighbors.emplace_back(kv.first, kv.second);
    }

    Index keep = cfg.k;
    if (static_cast<size_t>(keep) > neighbors.size()) {
      keep = static_cast<Index>(neighbors.size());
    }
    auto cmp = [](const auto& a, const auto& b) {
      if (a.second != b.second) {
        return a.second > b.second;
      }
      return a.first < b.first;
    };
    std::partial_sort(neighbors.begin(), neighbors.begin() + keep, neighbors.end(), cmp);

    for (Index i = 0; i < keep; ++i) {
      Index h = neighbors[static_cast<size_t>(i)].first;
      double w = neighbors[static_cast<size_t>(i)].second * cfg.weight;
      EdgeKey key = CanonicalEdge(g, h);
      auto it = edge_weights.find(key);
      if (it == edge_weights.end() || w > it->second) {
        edge_weights[key] = w;
      }
    }
  }

  if (edge_weights.empty()) {
    return WeightedCsrGraph(n_groups);
  }

  std::vector<Index> row_counts(static_cast<size_t>(n_groups), 0);
  for (const auto& kv : edge_weights) {
    const Index g = kv.first.first;
    const Index h = kv.first.second;
    if (g == h) {
      continue;
    }
    row_counts[static_cast<size_t>(g)] += 1;
    row_counts[static_cast<size_t>(h)] += 1;
  }

  std::vector<Index> row_ptr_out(static_cast<size_t>(n_groups + 1), 0);
  for (Index g = 0; g < n_groups; ++g) {
    row_ptr_out[static_cast<size_t>(g + 1)] =
        row_ptr_out[static_cast<size_t>(g)] + row_counts[static_cast<size_t>(g)];
  }

  const Index nnz = row_ptr_out.back();
  std::vector<Index> col_idx_out(static_cast<size_t>(nnz));
  std::vector<Scalar> values(static_cast<size_t>(nnz));
  std::vector<Index> offsets = row_ptr_out;

  auto add_edge = [&](Index u, Index v, double w) {
    const Index pos = offsets[static_cast<size_t>(u)]++;
    col_idx_out[static_cast<size_t>(pos)] = v;
    values[static_cast<size_t>(pos)] = w;
  };

  for (const auto& kv : edge_weights) {
    const Index g = kv.first.first;
    const Index h = kv.first.second;
    if (g == h) {
      continue;
    }
    const double w = kv.second;
    add_edge(g, h, w);
    add_edge(h, g, w);
  }

  return WeightedCsrGraph::FromCsr(n_groups, std::move(row_ptr_out),
                                   std::move(col_idx_out), std::move(values));
}

struct PairAccumulator {
  Index overlap_count = 0;
  double weight_sum = 0.0;
};

WeightedCsrGraph GenerateOverlapThreshold(const GroupGraphConfig& cfg,
                                          const BipartiteCsr& bipartite,
                                          Index n_groups) {
  // Generate an undirected group graph from overlap ratio thresholding.
  if (cfg.overlap_ratio_threshold < 0.0) {
    throw std::invalid_argument("GroupGraphConfig overlap_ratio_threshold must be >= 0");
  }
  if (cfg.target_mean_degree < 0.0) {
    throw std::invalid_argument("GroupGraphConfig target_mean_degree must be >= 0");
  }
  if (n_groups <= 0) {
    return WeightedCsrGraph(0);
  }

  const auto& row_ptr = bipartite.row_ptr();
  const auto& col_idx = bipartite.col_idx();
  const auto& values = bipartite.values();
  const Index n_users = bipartite.num_users();

  std::vector<Index> member_counts(static_cast<size_t>(n_groups), 0);
  std::unordered_map<EdgeKey, PairAccumulator, EdgeKeyHash> pair_accum;

  std::vector<std::pair<Index, double>> groups;
  for (Index u = 0; u < n_users; ++u) {
    groups.clear();
    const Index start = row_ptr[static_cast<size_t>(u)];
    const Index end = row_ptr[static_cast<size_t>(u + 1)];
    groups.reserve(static_cast<size_t>(end - start));
    for (Index idx = start; idx < end; ++idx) {
      groups.emplace_back(col_idx[static_cast<size_t>(idx)],
                          values[static_cast<size_t>(idx)]);
    }
    if (groups.empty()) {
      continue;
    }

    // Consolidate duplicates per (u, g): count membership once and sum weights.
    std::sort(groups.begin(), groups.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    Index unique = 0;
    for (const auto& item : groups) {
      if (unique > 0 &&
          groups[static_cast<size_t>(unique - 1)].first == item.first) {
        groups[static_cast<size_t>(unique - 1)].second += item.second;
      } else {
        groups[static_cast<size_t>(unique)] = item;
        unique += 1;
      }
    }
    groups.resize(static_cast<size_t>(unique));
    if (groups.size() <= 1) {
      if (groups.size() == 1) {
        const Index g = groups[0].first;
        member_counts[static_cast<size_t>(g)] += 1;
      }
      continue;
    }

    for (const auto& item : groups) {
      member_counts[static_cast<size_t>(item.first)] += 1;
    }

    for (size_t i = 0; i + 1 < groups.size(); ++i) {
      const Index g1 = groups[i].first;
      const double w1 = groups[i].second;
      for (size_t j = i + 1; j < groups.size(); ++j) {
        const Index g2 = groups[j].first;
        const double w2 = groups[j].second;
        EdgeKey key = CanonicalEdge(g1, g2);
        auto& acc = pair_accum[key];
        acc.overlap_count += 1;
        acc.weight_sum += (w1 * w2);
      }
    }
  }

  if (pair_accum.empty()) {
    return WeightedCsrGraph(n_groups);
  }

  std::vector<std::tuple<Index, Index, double>> edges;
  edges.reserve(pair_accum.size());
  std::vector<Index> row_counts(static_cast<size_t>(n_groups), 0);
  double total_weight = 0.0;

  for (const auto& kv : pair_accum) {
    const Index g = kv.first.first;
    const Index h = kv.first.second;
    if (g == h) {
      continue;
    }
    const Index min_size = std::min(member_counts[static_cast<size_t>(g)],
                                    member_counts[static_cast<size_t>(h)]);
    if (min_size <= 0) {
      continue;
    }
    const double ratio =
        static_cast<double>(kv.second.overlap_count) / static_cast<double>(min_size);
    if (ratio < cfg.overlap_ratio_threshold) {
      continue;
    }
    edges.emplace_back(g, h, kv.second.weight_sum);
    row_counts[static_cast<size_t>(g)] += 1;
    row_counts[static_cast<size_t>(h)] += 1;
    total_weight += kv.second.weight_sum;
  }

  if (edges.empty()) {
    return WeightedCsrGraph(n_groups);
  }

  const double mean_degree =
      (2.0 * total_weight) / static_cast<double>(n_groups);
  const double scale = mean_degree > 0.0 ? (cfg.target_mean_degree / mean_degree) : 0.0;

  std::vector<Index> row_ptr_out(static_cast<size_t>(n_groups + 1), 0);
  for (Index g = 0; g < n_groups; ++g) {
    row_ptr_out[static_cast<size_t>(g + 1)] =
        row_ptr_out[static_cast<size_t>(g)] + row_counts[static_cast<size_t>(g)];
  }

  const Index nnz = row_ptr_out.back();
  std::vector<Index> col_idx_out(static_cast<size_t>(nnz));
  std::vector<Scalar> values_out(static_cast<size_t>(nnz));
  std::vector<Index> offsets = row_ptr_out;

  auto add_edge = [&](Index u, Index v, double w) {
    const Index pos = offsets[static_cast<size_t>(u)]++;
    col_idx_out[static_cast<size_t>(pos)] = v;
    values_out[static_cast<size_t>(pos)] = w;
  };

  for (const auto& e : edges) {
    const Index g = std::get<0>(e);
    const Index h = std::get<1>(e);
    const double w = std::get<2>(e) * scale;
    add_edge(g, h, w);
    add_edge(h, g, w);
  }

  return WeightedCsrGraph::FromCsr(
      n_groups, std::move(row_ptr_out), std::move(col_idx_out),
      std::move(values_out));
}

}  // namespace

WeightedCsrGraph GroupGraphGenerator::Generate(const GroupGraphConfig& cfg,
                                               const BipartiteCsr* bipartite) {
  // Dispatch to the configured group-graph generator.
  const Index n_groups = ResolveGroupCount(cfg, bipartite);

  switch (cfg.mode) {
    case GroupGraphMode::kNone:
      return GenerateNone(n_groups);
    case GroupGraphMode::kErdosRenyi:
      if (cfg.p < 0.0 || cfg.p > 1.0) {
        throw std::invalid_argument("GroupGraphConfig p must be in [0, 1]");
      }
      return GenerateErdosRenyi(cfg, n_groups);
    case GroupGraphMode::kOverlapKnn:
      if (!bipartite) {
        throw std::invalid_argument("OverlapKnn requires bipartite data");
      }
      return GenerateOverlapKnn(cfg, *bipartite, n_groups);
    case GroupGraphMode::kOverlapThreshold:
      if (!bipartite) {
        throw std::invalid_argument("OverlapThreshold requires bipartite data");
      }
      return GenerateOverlapThreshold(cfg, *bipartite, n_groups);
    default:
      throw std::invalid_argument("Unknown group graph mode");
  }
}

}  // namespace fj
