#pragma once

#include <string>

#include "fj/common/types.hpp"
#include "fj/experiment/method.hpp"

namespace fj {

struct ExperimentResult {
  Method method = Method::Schur;
  std::string tag;

  Index outer_iters = 0;
  Index inner_iters = 0;
  double inner_seconds = 0.0;
  double seconds = 0.0;

  double relative_residual = 0.0;
  double disagreement = 0.0;
  double internal_conflict = 0.0;
  double polarization = 0.0;
  double controversy = 0.0;
  double user_graph_weight = 0.0;

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
};

}  // namespace fj
