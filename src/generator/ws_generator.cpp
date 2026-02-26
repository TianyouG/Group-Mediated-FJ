#include "fj/generator/ws_generator.hpp"

#include <algorithm>
#include <random>
#include <stdexcept>
#include <vector>

namespace fj {
namespace {

void ValidateConfig(const WsConfig& cfg) {
  // Validate WS generator parameters.
  if (cfg.n <= 0) {
    throw std::invalid_argument("WsConfig n must be positive");
  }
  if (cfg.k < 0 || cfg.k >= cfg.n) {
    throw std::invalid_argument("WsConfig k must satisfy 0 <= k < n");
  }
  if (cfg.k % 2 != 0) {
    throw std::invalid_argument("WsConfig k must be even");
  }
  if (cfg.p < 0.0 || cfg.p > 1.0) {
    throw std::invalid_argument("WsConfig p must be in [0, 1]");
  }
  if (cfg.weight <= 0.0) {
    throw std::invalid_argument("WsConfig weight must be positive");
  }
}

bool HasNeighbor(const std::vector<Index>& neighbors, Index v) {
  // Check if v is in the neighbor list.
  return std::find(neighbors.begin(), neighbors.end(), v) != neighbors.end();
}

void RemoveNeighbor(std::vector<Index>& neighbors, Index v) {
  // Remove v from the neighbor list (order not preserved).
  for (size_t i = 0; i < neighbors.size(); ++i) {
    if (neighbors[i] == v) {
      neighbors[i] = neighbors.back();
      neighbors.pop_back();
      return;
    }
  }
}

}  // namespace

WeightedCsrGraph WsGenerator::Generate(const WsConfig& cfg) {
  // Generate an undirected Watts-Strogatz graph.
  ValidateConfig(cfg);

  const Index n = cfg.n;
  const Index k = cfg.k;
  const Index half = k / 2;

  if (k == 0) {
    return WeightedCsrGraph(n);
  }

  std::mt19937_64 rng(cfg.seed);
  std::uniform_real_distribution<double> prob(0.0, 1.0);
  std::uniform_int_distribution<Index> pick(0, n - 1);

  std::vector<std::vector<Index>> adj(static_cast<size_t>(n));

  // Start from a ring lattice with degree k.
  for (Index u = 0; u < n; ++u) {
    for (Index j = 1; j <= half; ++j) {
      Index v = (u + j) % n;
      adj[static_cast<size_t>(u)].push_back(v);
      adj[static_cast<size_t>(v)].push_back(u);
    }
  }

  // Rewire edges with probability p.
  for (Index u = 0; u < n; ++u) {
    for (Index j = 1; j <= half; ++j) {
      Index v = (u + j) % n;
      if (prob(rng) >= cfg.p) {
        continue;
      }

      RemoveNeighbor(adj[static_cast<size_t>(u)], v);
      RemoveNeighbor(adj[static_cast<size_t>(v)], u);

      if (adj[static_cast<size_t>(u)].size() >= static_cast<size_t>(n - 1)) {
        adj[static_cast<size_t>(u)].push_back(v);
        adj[static_cast<size_t>(v)].push_back(u);
        continue;
      }

      Index w = u;
      do {
        w = pick(rng);
      } while (w == u || HasNeighbor(adj[static_cast<size_t>(u)], w));

      adj[static_cast<size_t>(u)].push_back(w);
      adj[static_cast<size_t>(w)].push_back(u);
    }
  }

  std::vector<Index> row_ptr(static_cast<size_t>(n + 1), 0);
  for (Index u = 0; u < n; ++u) {
    row_ptr[static_cast<size_t>(u + 1)] =
        row_ptr[static_cast<size_t>(u)] +
        static_cast<Index>(adj[static_cast<size_t>(u)].size());
  }

  const Index nnz = row_ptr.back();
  std::vector<Index> col_idx(static_cast<size_t>(nnz));
  std::vector<Scalar> values(static_cast<size_t>(nnz));
  std::vector<Index> offsets = row_ptr;

  // Emit directed edges with uniform weight from adjacency lists.
  for (Index u = 0; u < n; ++u) {
    for (Index v : adj[static_cast<size_t>(u)]) {
      const Index pos = offsets[static_cast<size_t>(u)]++;
      col_idx[static_cast<size_t>(pos)] = v;
      values[static_cast<size_t>(pos)] = cfg.weight;
    }
  }

  return WeightedCsrGraph::FromCsr(n, std::move(row_ptr), std::move(col_idx),
                                   std::move(values));
}

}  // namespace fj
