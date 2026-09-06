#include "fj/baseline/pf_qe.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "fj/graph/bipartite_reverse_index.hpp"

namespace fj {
namespace {

struct SampledEdge {
  Index user;
  Index neighbor;
  Scalar weight;
  size_t user_position;
  size_t neighbor_position;
};

void ValidateInputs(const WeightedCsrGraph& user_graph,
                    const WeightedCsrGraph& group_graph,
                    const BipartiteCsr& bipartite,
                    const Vector& lambda_u, const Vector& lambda_g,
                    const Vector& s_u, const Vector& s_g,
                    const PfQeOptions& options) {
  // Validate dimensions and sampling controls before constructing walk state.
  const Index n_users = bipartite.num_users();
  const Index n_groups = bipartite.num_groups();
  if (user_graph.num_nodes() != n_users ||
      group_graph.num_nodes() != n_groups || lambda_u.size() != n_users ||
      lambda_g.size() != n_groups || s_u.size() != n_users ||
      s_g.size() != n_groups) {
    throw std::invalid_argument("PfQeEstimator input size mismatch");
  }
  if (n_users <= 0) {
    throw std::invalid_argument("PfQeEstimator requires at least one user");
  }
  if (options.sample_users <= 0 || options.sample_edges < 0 ||
      options.forest_samples <= 0 || options.max_walk_steps <= 0) {
    throw std::invalid_argument("PfQeEstimator sampling parameters are invalid");
  }
  if (options.forest_samples >=
      static_cast<Index>(std::numeric_limits<std::uint32_t>::max())) {
    throw std::invalid_argument("PfQeEstimator forest_samples is too large");
  }
  if ((lambda_u.size() > 0 && lambda_u.minCoeff() < 0.0) ||
      (lambda_g.size() > 0 && lambda_g.minCoeff() < 0.0)) {
    throw std::invalid_argument("PfQeEstimator lambda values must be nonnegative");
  }
}

void ValidateWeights(const std::vector<Scalar>& values,
                     const char* graph_name) {
  // Random-walk transition probabilities require nonnegative edge weights.
  for (Scalar weight : values) {
    if (weight < 0.0) {
      throw std::invalid_argument(std::string(graph_name) +
                                  " weights must be nonnegative");
    }
  }
}

std::vector<Index> SampleUsersWithoutReplacement(Index population,
                                                 Index requested,
                                                 std::mt19937_64& rng) {
  // Floyd's algorithm samples a small subset without allocating population space.
  const Index count = std::min(population, requested);
  std::unordered_set<Index> selected;
  selected.reserve(static_cast<size_t>(count * 2 + 1));
  for (Index value = population - count; value < population; ++value) {
    std::uniform_int_distribution<Index> distribution(0, value);
    const Index candidate = distribution(rng);
    if (!selected.insert(candidate).second) {
      selected.insert(value);
    }
  }
  std::vector<Index> users(selected.begin(), selected.end());
  std::sort(users.begin(), users.end());
  return users;
}

class WeightedTwoLayerWalk {
 public:
  // Prepare degree vectors and the reverse UG view used by random walks.
  WeightedTwoLayerWalk(const WeightedCsrGraph& user_graph,
                       const WeightedCsrGraph& group_graph,
                       const BipartiteCsr& bipartite,
                       const Vector& lambda_u, const Vector& lambda_g,
                       const Vector& s_u, const Vector& s_g)
      : n_users_(bipartite.num_users()),
        n_groups_(bipartite.num_groups()),
        user_graph_(user_graph),
        group_graph_(group_graph),
        bipartite_(bipartite),
        reverse_bipartite_(bipartite),
        lambda_u_(lambda_u),
        lambda_g_(lambda_g),
        s_u_(s_u),
        s_g_(s_g),
        degree_u_(bipartite.user_degree()),
        degree_g_(bipartite.group_degree()) {
    if (user_graph_.nnz() > 0) {
      degree_u_ += user_graph_.degree();
    }
    if (group_graph_.nnz() > 0) {
      degree_g_ += group_graph_.degree();
    }
  }

  // Return total number of user and group nodes.
  Index num_nodes() const { return n_users_ + n_groups_; }

  // Return the internal opinion associated with an absorbing root.
  double InternalOpinion(Index node) const {
    return node < n_users_ ? s_u_[node] : s_g_[node - n_users_];
  }

  // Draw either absorption at the current node or one weighted neighbor.
  Index DrawNext(Index node, std::mt19937_64& rng, bool& absorbed) const {
    const bool is_user = node < n_users_;
    const Index local = is_user ? node : node - n_users_;
    const double lambda = is_user ? lambda_u_[local] : lambda_g_[local];
    const double degree = is_user ? degree_u_[local] : degree_g_[local];
    const double total = lambda + degree;
    if (total <= 0.0) {
      throw std::runtime_error(
          "PF-QE walk reached an isolated node without anchoring");
    }

    std::uniform_real_distribution<double> distribution(0.0, total);
    double target = distribution(rng);
    if (target < lambda) {
      absorbed = true;
      return node;
    }
    absorbed = false;
    target -= lambda;

    Index fallback = -1;
    if (is_user) {
      const auto& graph_rows = user_graph_.row_ptr();
      const auto& graph_cols = user_graph_.col_idx();
      const auto& graph_values = user_graph_.values();
      const Index graph_begin = graph_rows[static_cast<size_t>(local)];
      const Index graph_end = graph_rows[static_cast<size_t>(local + 1)];
      for (Index idx = graph_begin; idx < graph_end; ++idx) {
        const double weight = graph_values[static_cast<size_t>(idx)];
        if (weight > 0.0) {
          fallback = graph_cols[static_cast<size_t>(idx)];
        }
        if (target < weight) {
          return graph_cols[static_cast<size_t>(idx)];
        }
        target -= weight;
      }

      const auto& ug_rows = bipartite_.row_ptr();
      const auto& ug_cols = bipartite_.col_idx();
      const auto& ug_values = bipartite_.values();
      const Index ug_begin = ug_rows[static_cast<size_t>(local)];
      const Index ug_end = ug_rows[static_cast<size_t>(local + 1)];
      for (Index idx = ug_begin; idx < ug_end; ++idx) {
        const double weight = ug_values[static_cast<size_t>(idx)];
        if (weight > 0.0) {
          fallback = n_users_ + ug_cols[static_cast<size_t>(idx)];
        }
        if (target < weight) {
          return n_users_ + ug_cols[static_cast<size_t>(idx)];
        }
        target -= weight;
      }
    } else {
      const auto& graph_rows = group_graph_.row_ptr();
      const auto& graph_cols = group_graph_.col_idx();
      const auto& graph_values = group_graph_.values();
      const Index graph_begin = graph_rows[static_cast<size_t>(local)];
      const Index graph_end = graph_rows[static_cast<size_t>(local + 1)];
      for (Index idx = graph_begin; idx < graph_end; ++idx) {
        const double weight = graph_values[static_cast<size_t>(idx)];
        if (weight > 0.0) {
          fallback = n_users_ + graph_cols[static_cast<size_t>(idx)];
        }
        if (target < weight) {
          return n_users_ + graph_cols[static_cast<size_t>(idx)];
        }
        target -= weight;
      }

      const auto& gu_rows = reverse_bipartite_.row_ptr();
      const auto& gu_users = reverse_bipartite_.user_idx();
      const auto& gu_values = reverse_bipartite_.values();
      const Index gu_begin = gu_rows[static_cast<size_t>(local)];
      const Index gu_end = gu_rows[static_cast<size_t>(local + 1)];
      for (Index idx = gu_begin; idx < gu_end; ++idx) {
        const double weight = gu_values[static_cast<size_t>(idx)];
        if (weight > 0.0) {
          fallback = gu_users[static_cast<size_t>(idx)];
        }
        if (target < weight) {
          return gu_users[static_cast<size_t>(idx)];
        }
        target -= weight;
      }
    }

    if (fallback < 0) {
      throw std::runtime_error("PF-QE failed to draw a positive-weight neighbor");
    }
    return fallback;
  }

 private:
  Index n_users_;
  Index n_groups_;
  const WeightedCsrGraph& user_graph_;
  const WeightedCsrGraph& group_graph_;
  const BipartiteCsr& bipartite_;
  BipartiteReverseIndex reverse_bipartite_;
  const Vector& lambda_u_;
  const Vector& lambda_g_;
  const Vector& s_u_;
  const Vector& s_g_;
  Vector degree_u_;
  Vector degree_g_;
};

size_t AddTarget(Index user, std::vector<Index>& targets,
                 std::unordered_map<Index, size_t>& positions) {
  // Insert a target user once and return its position in the estimate vector.
  const auto found = positions.find(user);
  if (found != positions.end()) {
    return found->second;
  }
  const size_t position = targets.size();
  targets.push_back(user);
  positions.emplace(user, position);
  return position;
}

}  // namespace

PfQeResult PfQeEstimator::Estimate(const WeightedCsrGraph& user_graph,
                                   const WeightedCsrGraph& group_graph,
                                   const BipartiteCsr& bipartite,
                                   const Vector& lambda_u,
                                   const Vector& lambda_g,
                                   const Vector& s_u, const Vector& s_g,
                                   const PfQeOptions& options) {
  ValidateInputs(user_graph, group_graph, bipartite, lambda_u, lambda_g,
                 s_u, s_g, options);
  ValidateWeights(user_graph.values(), "User graph");
  ValidateWeights(group_graph.values(), "Group graph");
  ValidateWeights(bipartite.values(), "Bipartite graph");

  const Index n_users = bipartite.num_users();
  std::mt19937_64 rng(options.seed);
  std::vector<Index> metric_users = SampleUsersWithoutReplacement(
      n_users, options.sample_users, rng);

  std::vector<Index> targets;
  targets.reserve(metric_users.size() +
                  static_cast<size_t>(2 * options.sample_edges));
  std::unordered_map<Index, size_t> target_positions;
  target_positions.reserve(targets.capacity() * 2 + 1);
  for (Index user : metric_users) {
    AddTarget(user, targets, target_positions);
  }

  // Sample directed CSR entries; the final factor one-half removes symmetry.
  std::vector<SampledEdge> sampled_edges;
  const Index user_nnz = user_graph.nnz();
  if (user_nnz > 0 && options.sample_edges > 0) {
    sampled_edges.reserve(static_cast<size_t>(options.sample_edges));
    std::uniform_int_distribution<Index> edge_distribution(0, user_nnz - 1);
    const auto& rows = user_graph.row_ptr();
    const auto& cols = user_graph.col_idx();
    const auto& values = user_graph.values();
    for (Index sample = 0; sample < options.sample_edges; ++sample) {
      const Index edge_index = edge_distribution(rng);
      const auto upper = std::upper_bound(rows.begin(), rows.end(), edge_index);
      const Index user = static_cast<Index>(upper - rows.begin()) - 1;
      const Index neighbor = cols[static_cast<size_t>(edge_index)];
      const size_t user_position =
          AddTarget(user, targets, target_positions);
      const size_t neighbor_position =
          AddTarget(neighbor, targets, target_positions);
      sampled_edges.push_back(
          {user, neighbor, values[static_cast<size_t>(edge_index)],
           user_position, neighbor_position});
    }
  }

  WeightedTwoLayerWalk walk_graph(user_graph, group_graph, bipartite,
                                  lambda_u, lambda_g, s_u, s_g);
  const Index total_nodes = walk_graph.num_nodes();
  std::vector<std::uint32_t> forest_stamp(static_cast<size_t>(total_nodes), 0);
  std::vector<Index> root(static_cast<size_t>(total_nodes), -1);
  std::vector<Index> next(static_cast<size_t>(total_nodes), -1);
  std::vector<double> estimates(targets.size(), 0.0);
  Index total_walk_steps = 0;

  for (Index sample = 0; sample < options.forest_samples; ++sample) {
    const std::uint32_t stamp = static_cast<std::uint32_t>(sample + 1);
    for (size_t target_position = 0; target_position < targets.size();
         ++target_position) {
      const Index start = targets[target_position];
      Index node = start;
      Index current_walk_steps = 0;

      // Store last exits until the walk is absorbed or reaches this forest.
      while (forest_stamp[static_cast<size_t>(node)] != stamp) {
        if (current_walk_steps >= options.max_walk_steps) {
          throw std::runtime_error("PF-QE exceeded pf_max_walk_steps");
        }
        bool absorbed = false;
        const Index next_node = walk_graph.DrawNext(node, rng, absorbed);
        current_walk_steps += 1;
        total_walk_steps += 1;
        if (absorbed) {
          forest_stamp[static_cast<size_t>(node)] = stamp;
          root[static_cast<size_t>(node)] = node;
          next[static_cast<size_t>(node)] = -1;
          break;
        }
        next[static_cast<size_t>(node)] = next_node;
        node = next_node;
      }

      const Index absorbing_root = root[static_cast<size_t>(node)];
      if (absorbing_root < 0) {
        throw std::runtime_error("PF-QE forest root is unavailable");
      }

      // Following last-exit pointers performs chronological loop erasure.
      node = start;
      while (forest_stamp[static_cast<size_t>(node)] != stamp) {
        const Index next_node = next[static_cast<size_t>(node)];
        if (next_node < 0 || next_node >= total_nodes) {
          throw std::runtime_error("PF-QE encountered an invalid forest path");
        }
        forest_stamp[static_cast<size_t>(node)] = stamp;
        root[static_cast<size_t>(node)] = absorbing_root;
        node = next_node;
      }

      estimates[target_position] +=
          walk_graph.InternalOpinion(absorbing_root);
    }
  }

  const double inv_forests = 1.0 / static_cast<double>(options.forest_samples);
  for (double& estimate : estimates) {
    estimate *= inv_forests;
  }

  PfQeResult result;
  result.sampled_users = static_cast<Index>(metric_users.size());
  result.sampled_edges = static_cast<Index>(sampled_edges.size());
  result.forest_samples = options.forest_samples;
  result.walk_steps = total_walk_steps;

  const double node_scale =
      static_cast<double>(n_users) / static_cast<double>(metric_users.size());
  double mean = 0.0;
  for (size_t pos = 0; pos < metric_users.size(); ++pos) {
    mean += estimates[pos];
  }
  mean /= static_cast<double>(metric_users.size());

  for (size_t pos = 0; pos < metric_users.size(); ++pos) {
    const Index user = metric_users[pos];
    const double opinion = estimates[pos];
    result.controversy += opinion * opinion;
    const double conflict = opinion - s_u[user];
    result.internal_conflict += conflict * conflict;
    const double centered = opinion - mean;
    result.polarization += centered * centered;
  }
  result.controversy *= node_scale;
  result.internal_conflict *= node_scale;
  result.polarization *= node_scale;

  if (!sampled_edges.empty()) {
    double edge_sum = 0.0;
    for (const SampledEdge& edge : sampled_edges) {
      const double difference = estimates[edge.user_position] -
                                estimates[edge.neighbor_position];
      edge_sum += edge.weight * difference * difference;
    }
    result.disagreement =
        0.5 * static_cast<double>(user_nnz) * edge_sum /
        static_cast<double>(sampled_edges.size());
  }
  return result;
}

}  // namespace fj

