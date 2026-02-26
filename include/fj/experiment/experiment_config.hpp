#pragma once

#include <cstdint>
#include <string>

#include "fj/common/types.hpp"
#include "fj/experiment/method.hpp"

namespace fj {

struct ExperimentConfig {
  Method method = Method::Schur;

  Index n_users = 0;
  Index n_groups = 0;

  bool use_real_data = false;
  std::string user_graph_path;
  std::string group_graph_path;
  std::string bipartite_path;
  std::string user_graph_bin_path;
  std::string group_graph_bin_path;
  std::string bipartite_bin_path;
  std::string su_path;
  std::string sg_path;
  std::string lambda_u_path;
  std::string lambda_g_path;
  bool data_one_indexed = false;
  bool data_ignore_self_loops = true;
  bool user_graph_symmetrize = false;
  bool group_graph_symmetrize = false;

  double alpha = 2.0;
  Index s_min = 1;
  Index s_max = 1;
  double user_mean = 1.0;
  Index user_r_max = 1;
  bool ug_normalize_groups = true;
  std::string su_distribution = "uniform";
  double exp_lambda = 1.0;
  double power_alpha = 2.0;
  double power_min = 1.0;
  double power_max = 100.0;

  Index k_ws = 0;
  double p_ws = 0.0;
  double user_graph_weight = -1.0;

  bool enable_group_graph = false;
  double group_overlap_threshold = 0.5;
  double group_graph_target_mean_degree = 1.0;

  double lambda_user = 1.0;
  double lambda_group = 1e-4;

  Index outer_max_iters = 200;
  double outer_tol = 1e-8;
  Index inner_max_iters = 200;
  double inner_tol = 1e-8;

  bool clique_use_weights = false;
  bool schur_use_jacobi = true;
  bool full_system_use_jacobi = true;
  std::string schur_precond_path;
  std::string full_precond_path;
  Index direct_max_dim = 2000;

  uint64_t seed = 1;
  std::string tag;
};

}  // namespace fj
