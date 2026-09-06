#pragma once

#include <limits>
#include <string>
#include <vector>

#include "fj/common/types.hpp"
#include "fj/experiment/method.hpp"

namespace fj {

struct ConvergenceTraceEntry {
  std::string level;
  Index solve_id = 0;
  Index iteration = 0;
  double relative_residual = 0.0;
  double seconds = 0.0;
};

struct ExperimentResult {
  Method method = Method::Schur;
  std::string tag;

  Index outer_iters = 0;
  Index inner_iters = 0;
  Index local_updates = 0;
  double inner_seconds = 0.0;
  double seconds = 0.0;

  double relative_residual = 0.0;
  double full_relative_residual =
      std::numeric_limits<double>::quiet_NaN();
  double disagreement = 0.0;
  double internal_conflict = 0.0;
  double polarization = 0.0;
  double controversy = 0.0;
  double user_graph_weight = 0.0;
  double node_mean_absolute_error =
      std::numeric_limits<double>::quiet_NaN();
  double node_max_absolute_error =
      std::numeric_limits<double>::quiet_NaN();
  double node_relative_l2_error =
      std::numeric_limits<double>::quiet_NaN();
  double peak_memory_mb = std::numeric_limits<double>::quiet_NaN();
  double inner_alpha_max = 0.0;
  double inner_condition_bound = 1.0;

  Index nnz_w = 0;
  double max_group_size = 0.0;
  double sum_group_size_sq = 0.0;
  double user_cross_degree_mean = 0.0;
  double user_cross_degree_max = 0.0;
  double group_cross_degree_mean = 0.0;
  double group_cross_degree_max = 0.0;
  double user_graph_degree_mean = 0.0;
  double user_graph_degree_max = 0.0;
  double group_graph_degree_mean = 0.0;
  double group_graph_degree_max = 0.0;

  Index sampled_users = 0;
  Index sampled_edges = 0;
  Index forest_samples = 0;
  Index walk_steps = 0;

  Vector x_u;
  std::vector<ConvergenceTraceEntry> convergence_trace;
};

}  // namespace fj
