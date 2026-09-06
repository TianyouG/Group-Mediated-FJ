#include "fj/baseline/bli_solver.hpp"

#include <cmath>
#include <deque>
#include <stdexcept>
#include <vector>

#include "fj/common/timer.hpp"
#include "fj/graph/bipartite_reverse_index.hpp"

namespace fj {
namespace {

constexpr Index kRoundMarker = -1;

void ValidateInputs(const WeightedCsrGraph& user_graph,
                    const WeightedCsrGraph& group_graph,
                    const BipartiteCsr& bipartite,
                    const Vector& lambda_u, const Vector& lambda_g,
                    const Vector& b_u, const Vector& b_g,
                    const BliOptions& options) {
  // Validate dimensions and numerical parameters before allocating solver state.
  const Index n_users = bipartite.num_users();
  const Index n_groups = bipartite.num_groups();
  if (user_graph.num_nodes() != n_users ||
      group_graph.num_nodes() != n_groups || lambda_u.size() != n_users ||
      lambda_g.size() != n_groups || b_u.size() != n_users ||
      b_g.size() != n_groups) {
    throw std::invalid_argument("BliSolver input size mismatch");
  }
  if (options.max_rounds <= 0) {
    throw std::invalid_argument("BliSolver max_rounds must be positive");
  }
  if (options.max_updates < 0) {
    throw std::invalid_argument("BliSolver max_updates must be nonnegative");
  }
  if (options.tolerance <= 0.0) {
    throw std::invalid_argument("BliSolver tolerance must be positive");
  }
  if (!(options.omega > 0.0 && options.omega < 2.0)) {
    throw std::invalid_argument("BliSolver omega must be in (0, 2)");
  }
  if ((lambda_u.size() > 0 && lambda_u.minCoeff() < 0.0) ||
      (lambda_g.size() > 0 && lambda_g.minCoeff() < 0.0)) {
    throw std::invalid_argument("BliSolver lambda values must be nonnegative");
  }
}

void ValidateWeights(const std::vector<Scalar>& values,
                     const char* graph_name) {
  // The Laplacian and probabilistic interpretations require nonnegative edges.
  for (Scalar weight : values) {
    if (weight < 0.0) {
      throw std::invalid_argument(std::string(graph_name) +
                                  " weights must be nonnegative");
    }
  }
}

}  // namespace

BliResult BliSolver::Solve(const WeightedCsrGraph& user_graph,
                           const WeightedCsrGraph& group_graph,
                           const BipartiteCsr& bipartite,
                           const Vector& lambda_u, const Vector& lambda_g,
                           const Vector& b_u, const Vector& b_g,
                           const BliOptions& options) {
  ValidateInputs(user_graph, group_graph, bipartite, lambda_u, lambda_g,
                 b_u, b_g, options);
  ValidateWeights(user_graph.values(), "User graph");
  ValidateWeights(group_graph.values(), "Group graph");
  ValidateWeights(bipartite.values(), "Bipartite graph");

  const Index n_users = bipartite.num_users();
  const Index n_groups = bipartite.num_groups();
  const Index total_nodes = n_users + n_groups;

  BliResult result;
  result.x_u = Vector::Zero(n_users);
  result.x_g = Vector::Zero(n_groups);
  Vector residual_u = b_u;
  Vector residual_g = b_g;
  Timer timer;

  const double b_norm =
      std::sqrt(b_u.squaredNorm() + b_g.squaredNorm());
  if (b_norm == 0.0 || total_nodes == 0) {
    result.full_relative_residual = 0.0;
    result.converged = true;
    if (options.record_history) {
      result.residual_history.push_back({0, 0.0, timer.ElapsedSeconds()});
    }
    return result;
  }
  Vector diagonal_u = lambda_u + bipartite.user_degree();
  if (user_graph.nnz() > 0) {
    diagonal_u += user_graph.degree();
  }
  Vector diagonal_g = lambda_g + bipartite.group_degree();
  if (group_graph.nnz() > 0) {
    diagonal_g += group_graph.degree();
  }

  // Build the reverse index only for local algorithms that need group pushes.
  BipartiteReverseIndex reverse_bipartite(bipartite);

  const double component_threshold =
      options.tolerance * b_norm /
      std::sqrt(static_cast<double>(total_nodes));
  std::deque<Index> queue;
  std::vector<unsigned char> in_queue(static_cast<size_t>(total_nodes), 0);

  auto residual_value = [&](Index node) -> double {
    return node < n_users ? residual_u[node] : residual_g[node - n_users];
  };
  auto add_user_residual = [&](Index user, double increment) {
    residual_u[user] += increment;
  };
  auto add_group_residual = [&](Index group, double increment) {
    residual_g[group] += increment;
  };
  auto record_round = [&]() {
    if (options.record_history) {
      const double residual_norm = std::sqrt(
          residual_u.squaredNorm() + residual_g.squaredNorm());
      result.residual_history.push_back(
          {result.rounds, residual_norm / b_norm, timer.ElapsedSeconds()});
    }
  };
  auto activate = [&](Index node) {
    if (std::abs(residual_value(node)) <= component_threshold) {
      return;
    }
    auto& flag = in_queue[static_cast<size_t>(node)];
    if (flag == 0) {
      flag = 1;
      queue.push_back(node);
    }
  };

  queue.push_back(kRoundMarker);
  for (Index user = 0; user < n_users; ++user) {
    activate(user);
  }
  for (Index group = 0; group < n_groups; ++group) {
    activate(n_users + group);
  }

  const auto& user_rows = user_graph.row_ptr();
  const auto& user_cols = user_graph.col_idx();
  const auto& user_weights = user_graph.values();
  const auto& group_rows = group_graph.row_ptr();
  const auto& group_cols = group_graph.col_idx();
  const auto& group_weights = group_graph.values();
  const auto& ug_rows = bipartite.row_ptr();
  const auto& ug_cols = bipartite.col_idx();
  const auto& ug_weights = bipartite.values();
  const auto& gu_rows = reverse_bipartite.row_ptr();
  const auto& gu_users = reverse_bipartite.user_idx();
  const auto& gu_weights = reverse_bipartite.values();

  bool stopped_by_limit = false;
  while (!queue.empty()) {
    const Index node = queue.front();
    queue.pop_front();

    if (node == kRoundMarker) {
      record_round();
      if (queue.empty()) {
        break;
      }
      if (result.rounds >= options.max_rounds) {
        stopped_by_limit = true;
        break;
      }
      result.rounds += 1;
      queue.push_back(kRoundMarker);
      continue;
    }

    in_queue[static_cast<size_t>(node)] = 0;
    if (std::abs(residual_value(node)) <= component_threshold) {
      continue;
    }
    if (options.max_updates > 0 && result.updates >= options.max_updates) {
      stopped_by_limit = true;
      break;
    }

    if (node < n_users) {
      // Push one user residual through user-user and user-group edges.
      const Index user = node;
      const double diagonal = diagonal_u[user];
      if (diagonal <= 0.0) {
        throw std::runtime_error(
            "BLI encountered an active user with zero diagonal");
      }
      const double old_residual = residual_u[user];
      const double delta = options.omega * old_residual / diagonal;
      result.x_u[user] += delta;
      add_user_residual(user, -diagonal * delta);

      const Index user_begin = user_rows[static_cast<size_t>(user)];
      const Index user_end = user_rows[static_cast<size_t>(user + 1)];
      for (Index idx = user_begin; idx < user_end; ++idx) {
        const Index neighbor = user_cols[static_cast<size_t>(idx)];
        add_user_residual(neighbor,
                          user_weights[static_cast<size_t>(idx)] * delta);
        activate(neighbor);
      }

      const Index ug_begin = ug_rows[static_cast<size_t>(user)];
      const Index ug_end = ug_rows[static_cast<size_t>(user + 1)];
      for (Index idx = ug_begin; idx < ug_end; ++idx) {
        const Index group = ug_cols[static_cast<size_t>(idx)];
        add_group_residual(group,
                           ug_weights[static_cast<size_t>(idx)] * delta);
        activate(n_users + group);
      }
      activate(user);
    } else {
      // Push one group residual through group-group and group-user edges.
      const Index group = node - n_users;
      const double diagonal = diagonal_g[group];
      if (diagonal <= 0.0) {
        throw std::runtime_error(
            "BLI encountered an active group with zero diagonal");
      }
      const double old_residual = residual_g[group];
      const double delta = options.omega * old_residual / diagonal;
      result.x_g[group] += delta;
      add_group_residual(group, -diagonal * delta);

      const Index group_begin = group_rows[static_cast<size_t>(group)];
      const Index group_end = group_rows[static_cast<size_t>(group + 1)];
      for (Index idx = group_begin; idx < group_end; ++idx) {
        const Index neighbor = group_cols[static_cast<size_t>(idx)];
        add_group_residual(neighbor,
                           group_weights[static_cast<size_t>(idx)] * delta);
        activate(n_users + neighbor);
      }

      const Index gu_begin = gu_rows[static_cast<size_t>(group)];
      const Index gu_end = gu_rows[static_cast<size_t>(group + 1)];
      for (Index idx = gu_begin; idx < gu_end; ++idx) {
        const Index user = gu_users[static_cast<size_t>(idx)];
        add_user_residual(user,
                          gu_weights[static_cast<size_t>(idx)] * delta);
        activate(user);
      }
      activate(node);
    }
    result.updates += 1;
  }

  const double residual_norm =
      std::sqrt(residual_u.squaredNorm() + residual_g.squaredNorm());
  result.full_relative_residual = residual_norm / b_norm;
  result.converged = !stopped_by_limit &&
                     result.full_relative_residual <= options.tolerance;
  return result;
}

}  // namespace fj
