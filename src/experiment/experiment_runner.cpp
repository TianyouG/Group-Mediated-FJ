#include "fj/experiment/experiment_runner.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <stdexcept>
#include <vector>

#include "fj/experiment/experiment_instance.hpp"
#include "fj/experiment/methods.hpp"
#include "fj/generator/group_graph_generator.hpp"
#include "fj/generator/ug_generator.hpp"
#include "fj/generator/ws_generator.hpp"
#include "fj/io/binary_csr_io.hpp"
#include "fj/io/edge_list_reader.hpp"
#include "fj/io/vector_reader.hpp"

namespace fj {
namespace {

void ValidateConfig(const ExperimentConfig& cfg) {
  // Validate configuration based on input mode and solver requirements.
  if (cfg.lambda_user <= 0.0) {
    throw std::invalid_argument("ExperimentConfig lambda_user must be positive");
  }
  if (cfg.lambda_group < 0.0) {
    throw std::invalid_argument("ExperimentConfig lambda_group must be nonnegative");
  }
  if (cfg.user_graph_weight != -1.0 && cfg.user_graph_weight <= 0.0) {
    throw std::invalid_argument("ExperimentConfig user_graph_weight must be "
                                "positive or -1 for auto");
  }
  if (cfg.use_real_data) {
    if (cfg.bipartite_path.empty() && cfg.bipartite_bin_path.empty()) {
      throw std::invalid_argument("ExperimentConfig bipartite input required for real data");
    }
    if (!cfg.bipartite_path.empty() && !cfg.bipartite_bin_path.empty()) {
      throw std::invalid_argument("Provide only one of bipartite_path or bipartite_bin_path");
    }
    if (!cfg.user_graph_path.empty() && !cfg.user_graph_bin_path.empty()) {
      throw std::invalid_argument("Provide only one of user_graph_path or user_graph_bin_path");
    }
    if (!cfg.group_graph_path.empty() && !cfg.group_graph_bin_path.empty()) {
      throw std::invalid_argument("Provide only one of group_graph_path or group_graph_bin_path");
    }
    if (cfg.n_users < 0 || cfg.n_groups < 0) {
      throw std::invalid_argument("ExperimentConfig n_users/n_groups cannot be negative");
    }
    if ((!cfg.schur_precond_path.empty() || !cfg.full_precond_path.empty()) &&
        (!cfg.lambda_u_path.empty() || !cfg.lambda_g_path.empty())) {
      throw std::invalid_argument("Preconditioner files require constant lambda values");
    }
    return;
  }
  if (cfg.n_users <= 0 || cfg.n_groups <= 0) {
    throw std::invalid_argument("ExperimentConfig n_users/n_groups must be positive");
  }
  if (cfg.k_ws < 0 || cfg.k_ws >= cfg.n_users || (cfg.k_ws % 2 != 0)) {
    throw std::invalid_argument("ExperimentConfig k_ws must be even and < n_users");
  }
  if (cfg.p_ws < 0.0 || cfg.p_ws > 1.0) {
    throw std::invalid_argument("ExperimentConfig p_ws must be in [0, 1]");
  }
  if (cfg.su_distribution != "uniform" && cfg.su_distribution != "exponential" &&
      cfg.su_distribution != "powerlaw") {
    throw std::invalid_argument(
        "ExperimentConfig su_distribution must be uniform|exponential|powerlaw");
  }
  if (cfg.su_distribution == "exponential" && cfg.exp_lambda <= 0.0) {
    throw std::invalid_argument("ExperimentConfig exp_lambda must be positive");
  }
  if (cfg.su_distribution == "powerlaw" && cfg.power_alpha <= 0.0) {
    throw std::invalid_argument("ExperimentConfig power_alpha must be positive");
  }
  if (cfg.su_distribution == "powerlaw" &&
      (cfg.power_min <= 0.0 || cfg.power_max <= cfg.power_min)) {
    throw std::invalid_argument(
        "ExperimentConfig power_min/power_max must satisfy 0 < min < max");
  }
  if (cfg.group_overlap_threshold < 0.0) {
    throw std::invalid_argument("ExperimentConfig group_overlap_threshold must be >= 0");
  }
  if (cfg.group_graph_target_mean_degree < 0.0) {
    throw std::invalid_argument(
        "ExperimentConfig group_graph_target_mean_degree must be >= 0");
  }
}

GroupGraphMode ResolveGroupMode(const ExperimentConfig& cfg) {
  // Determine which group graph generator to use.
  if (!cfg.enable_group_graph) {
    return GroupGraphMode::kNone;
  }
  return GroupGraphMode::kOverlapThreshold;
}

double ResolveUserGraphWeight(const ExperimentConfig& cfg) {
  // Resolve the synthetic user-graph edge weight.
  if (cfg.user_graph_weight >= 0.0) {
    return cfg.user_graph_weight;
  }
  if (cfg.k_ws <= 0) {
    return 1.0;
  }
  return 1.0 / static_cast<double>(cfg.k_ws);
}

double MeanEdgeWeight(const WeightedCsrGraph& graph) {
  // Compute the mean weight per stored edge (0 for empty graphs).
  const Index nnz = graph.nnz();
  if (nnz <= 0) {
    return 0.0;
  }
  const double total_weight = graph.total_weight();
  return total_weight / static_cast<double>(nnz);
}

Vector SampleSignedUserOpinions(Index n_users, uint64_t seed) {
  // Sample random +/- 1 opinions for users.
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  Vector s_u(n_users);
  for (Index i = 0; i < n_users; ++i) {
    s_u[i] = dist(rng) < 0.5 ? -1.0 : 1.0;
  }
  return s_u;
}

double SafeMax(const Vector& v) {
  return v.size() > 0 ? v.maxCoeff() : 0.0;
}

Vector SamplePowerLaw(std::mt19937_64& rng, Index n, double alpha,
                      double xmin, double xmax) {
  // Sample continuous power-law values in [xmin, xmax].
  if (alpha <= 0.0) {
    throw std::invalid_argument("power_alpha must be positive");
  }
  if (xmin <= 0.0 || xmax <= xmin) {
    throw std::invalid_argument("power range must satisfy 0 < min < max");
  }
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  Vector values(n);
  if (std::abs(alpha - 1.0) < 1e-12) {
    const double log_ratio = std::log(xmax / xmin);
    for (Index i = 0; i < n; ++i) {
      const double u = uni(rng);
      values[i] = xmin * std::exp(u * log_ratio);
    }
    return values;
  }

  const double a = 1.0 - alpha;
  const double xmin_a = std::pow(xmin, a);
  const double xmax_a = std::pow(xmax, a);
  for (Index i = 0; i < n; ++i) {
    const double u = uni(rng);
    values[i] = std::pow(u * (xmax_a - xmin_a) + xmin_a, 1.0 / a);
  }
  return values;
}

Vector SampleSyntheticUserOpinions(const ExperimentConfig& cfg,
                                   Index n_users) {
  // Sample synthetic s_u in [0, 1] with optional distribution choices.
  std::mt19937_64 rng(cfg.seed + 3);
  Vector s_u(n_users);

  if (cfg.su_distribution == "uniform") {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (Index i = 0; i < n_users; ++i) {
      s_u[i] = dist(rng);
    }
    return s_u;
  }

  if (cfg.su_distribution == "exponential") {
    if (cfg.exp_lambda <= 0.0) {
      throw std::invalid_argument("exp_lambda must be positive for exponential s_u");
    }
    std::exponential_distribution<double> dist(cfg.exp_lambda);
    for (Index i = 0; i < n_users; ++i) {
      s_u[i] = dist(rng);
    }
    const double mx = SafeMax(s_u);
    if (mx > 0.0) {
      s_u /= mx;
    }
    return s_u;
  }

  if (cfg.su_distribution == "powerlaw") {
    s_u = SamplePowerLaw(rng, n_users, cfg.power_alpha, cfg.power_min, cfg.power_max);
    const double mx = SafeMax(s_u);
    if (mx > 0.0) {
      s_u /= mx;
    }
    return s_u;
  }

  throw std::invalid_argument("Unknown su_distribution: " + cfg.su_distribution);
}

Vector ComputeGroupOpinionMeans(const BipartiteCsr& bipartite,
                                const Vector& s_u) {
  // Compute s_g as the unweighted mean of member s_u values for each group.
  const Index n_users = bipartite.num_users();
  const Index n_groups = bipartite.num_groups();
  const auto& row_ptr = bipartite.row_ptr();
  const auto& col_idx = bipartite.col_idx();

  Vector sum = Vector::Zero(n_groups);
  std::vector<Index> counts(static_cast<size_t>(n_groups), 0);

  for (Index u = 0; u < n_users; ++u) {
    const Index start = row_ptr[static_cast<size_t>(u)];
    const Index end = row_ptr[static_cast<size_t>(u + 1)];
    for (Index idx = start; idx < end; ++idx) {
      const Index g = col_idx[static_cast<size_t>(idx)];
      sum[g] += s_u[u];
      counts[static_cast<size_t>(g)] += 1;
    }
  }

  Vector s_g = Vector::Zero(n_groups);
  for (Index g = 0; g < n_groups; ++g) {
    const Index c = counts[static_cast<size_t>(g)];
    if (c > 0) {
      s_g[g] = sum[g] / static_cast<double>(c);
    }
  }
  return s_g;
}

double MeanOrZero(const Vector& v) {
  // Compute mean or return 0 for empty vectors.
  return v.size() > 0 ? v.mean() : 0.0;
}

double MaxOrZero(const Vector& v) {
  // Compute max or return 0 for empty vectors.
  return v.size() > 0 ? v.maxCoeff() : 0.0;
}

double SumSquares(const Vector& v) {
  // Compute sum of squares or return 0 for empty vectors.
  return v.size() > 0 ? v.squaredNorm() : 0.0;
}

void FillDiagnostics(const ExperimentInstance& instance,
                     ExperimentResult& result) {
  // Populate diagnostic metrics derived from the instance.
  result.nnz_w = instance.bipartite.nnz();

  result.max_group_size = MaxOrZero(instance.group_sizes);
  result.sum_group_size_sq = SumSquares(instance.group_sizes);

  const Vector& user_cross = instance.bipartite.user_degree();
  const Vector& group_cross = instance.bipartite.group_degree();
  result.user_cross_degree_mean = MeanOrZero(user_cross);
  result.user_cross_degree_max = MaxOrZero(user_cross);
  result.group_cross_degree_mean = MeanOrZero(group_cross);
  result.group_cross_degree_max = MaxOrZero(group_cross);

  const Vector& user_graph_deg = instance.user_graph.degree();
  const Vector& group_graph_deg = instance.group_graph.degree();
  result.user_graph_degree_mean = MeanOrZero(user_graph_deg);
  result.user_graph_degree_max = MaxOrZero(user_graph_deg);
  result.group_graph_degree_mean = MeanOrZero(group_graph_deg);
  result.group_graph_degree_max = MaxOrZero(group_graph_deg);
}

ExperimentInstance BuildInstanceFromFiles(const ExperimentConfig& config) {
  // Build an experiment instance from file-based graph inputs.
  EdgeListOptions ug_opts;
  ug_opts.one_indexed = config.data_one_indexed;
  ug_opts.ignore_self_loops = config.data_ignore_self_loops;
  ug_opts.symmetrize = false;

  BipartiteCsr bipartite =
      config.bipartite_bin_path.empty()
          ? EdgeListReader::ReadBipartite(config.bipartite_path, config.n_users,
                                          config.n_groups, ug_opts)
          : BinaryCsrIO::ReadBipartite(config.bipartite_bin_path, config.n_users,
                                       config.n_groups);
  const Index n_users = bipartite.num_users();
  const Index n_groups = bipartite.num_groups();

  EdgeListOptions user_opts = ug_opts;
  user_opts.symmetrize = config.user_graph_symmetrize;
  WeightedCsrGraph user_graph =
      config.user_graph_bin_path.empty()
          ? (config.user_graph_path.empty()
                 ? WeightedCsrGraph(n_users)
                 : EdgeListReader::ReadWeightedGraph(config.user_graph_path,
                                                     n_users, user_opts))
          : BinaryCsrIO::ReadWeightedGraph(config.user_graph_bin_path, n_users);

  EdgeListOptions group_opts = ug_opts;
  group_opts.symmetrize = config.group_graph_symmetrize;
  WeightedCsrGraph group_graph =
      config.group_graph_bin_path.empty()
          ? (config.group_graph_path.empty()
                 ? WeightedCsrGraph(n_groups)
                 : EdgeListReader::ReadWeightedGraph(config.group_graph_path,
                                                     n_groups, group_opts))
          : BinaryCsrIO::ReadWeightedGraph(config.group_graph_bin_path, n_groups);

  Vector s_u = config.su_path.empty()
                   ? SampleSignedUserOpinions(n_users, config.seed + 3)
                   : VectorReader::ReadVector(config.su_path, n_users);
  Vector s_g = config.sg_path.empty()
                   ? Vector::Zero(n_groups)
                   : VectorReader::ReadVector(config.sg_path, n_groups);

  Vector lambda_u =
      config.lambda_u_path.empty()
          ? Vector::Constant(n_users, config.lambda_user)
          : VectorReader::ReadVector(config.lambda_u_path, n_users);
  Vector lambda_g =
      config.lambda_g_path.empty()
          ? Vector::Constant(n_groups, config.lambda_group)
          : VectorReader::ReadVector(config.lambda_g_path, n_groups);

  if (lambda_u.minCoeff() < 0.0 || lambda_g.minCoeff() < 0.0) {
    throw std::invalid_argument("lambda values must be nonnegative");
  }

  Vector b_u = lambda_u.cwiseProduct(s_u);
  Vector b_g = lambda_g.cwiseProduct(s_g);

  ExperimentInstance instance;
  instance.user_graph = std::move(user_graph);
  instance.group_graph = std::move(group_graph);
  instance.bipartite = std::move(bipartite);
  instance.group_sizes = instance.bipartite.group_degree();
  instance.user_degrees = instance.bipartite.user_degree();
  instance.s_u = std::move(s_u);
  instance.s_g = std::move(s_g);
  instance.lambda_u = std::move(lambda_u);
  instance.lambda_g = std::move(lambda_g);
  instance.b_u = std::move(b_u);
  instance.b_g = std::move(b_g);
  return instance;
}

ExperimentInstance BuildInstance(const ExperimentConfig& config) {
  // Build an experiment instance from synthetic generators.
  if (config.use_real_data) {
    return BuildInstanceFromFiles(config);
  }
  UgConfig ug_cfg;
  ug_cfg.n_users = config.n_users;
  ug_cfg.n_groups = config.n_groups;
  ug_cfg.alpha = config.alpha;
  ug_cfg.s_min = config.s_min;
  ug_cfg.s_max = config.s_max;
  ug_cfg.user_mean = config.user_mean;
  ug_cfg.user_r_max = config.user_r_max;
  ug_cfg.seed = config.seed;

  UgResult ug = UgGenerator::Generate(ug_cfg);
  if (config.ug_normalize_groups) {
    // Normalize each group column so that sum_u w_{u,g} = 1.0.
    const Index n_users = ug.bipartite.num_users();
    const Index n_groups = ug.bipartite.num_groups();
    Vector col_sum = Vector::Zero(n_groups);
    const auto& row_ptr = ug.bipartite.row_ptr();
    const auto& col_idx = ug.bipartite.col_idx();
    const auto& values = ug.bipartite.values();
    for (Index u = 0; u < n_users; ++u) {
      const Index start = row_ptr[static_cast<size_t>(u)];
      const Index end = row_ptr[static_cast<size_t>(u + 1)];
      for (Index idx = start; idx < end; ++idx) {
        const Index g = col_idx[static_cast<size_t>(idx)];
        col_sum[g] += values[static_cast<size_t>(idx)];
      }
    }
    auto& mutable_values = ug.bipartite.values_mut();
    for (Index u = 0; u < n_users; ++u) {
      const Index start = row_ptr[static_cast<size_t>(u)];
      const Index end = row_ptr[static_cast<size_t>(u + 1)];
      for (Index idx = start; idx < end; ++idx) {
        const Index g = col_idx[static_cast<size_t>(idx)];
        const double denom = col_sum[g];
        if (denom > 0.0) {
          mutable_values[static_cast<size_t>(idx)] /= denom;
        }
      }
    }
  }

  WsConfig ws_cfg;
  ws_cfg.n = config.n_users;
  ws_cfg.k = config.k_ws;
  ws_cfg.p = config.p_ws;
  ws_cfg.weight = ResolveUserGraphWeight(config);
  ws_cfg.seed = config.seed + 1;

  WeightedCsrGraph user_graph = WsGenerator::Generate(ws_cfg);

  GroupGraphConfig gg_cfg;
  gg_cfg.mode = ResolveGroupMode(config);
  gg_cfg.n_groups = config.n_groups;
  gg_cfg.overlap_ratio_threshold = config.group_overlap_threshold;
  gg_cfg.target_mean_degree = config.group_graph_target_mean_degree;
  gg_cfg.seed = config.seed + 2;

  WeightedCsrGraph group_graph = GroupGraphGenerator::Generate(gg_cfg, &ug.bipartite);

  Vector s_u = SampleSyntheticUserOpinions(config, config.n_users);
  Vector s_g = ComputeGroupOpinionMeans(ug.bipartite, s_u);

  Vector lambda_u = Vector::Constant(config.n_users, config.lambda_user);
  Vector lambda_g = Vector::Constant(config.n_groups, config.lambda_group);

  Vector b_u = lambda_u.cwiseProduct(s_u);
  Vector b_g = lambda_g.cwiseProduct(s_g);

  ExperimentInstance instance;
  instance.user_graph = std::move(user_graph);
  instance.group_graph = std::move(group_graph);
  instance.bipartite = std::move(ug.bipartite);
  instance.group_sizes = std::move(ug.group_sizes);
  instance.user_degrees = std::move(ug.user_degrees);
  instance.s_u = std::move(s_u);
  instance.s_g = std::move(s_g);
  instance.lambda_u = std::move(lambda_u);
  instance.lambda_g = std::move(lambda_g);
  instance.b_u = std::move(b_u);
  instance.b_g = std::move(b_g);
  return instance;
}

}  // namespace

// Store the experiment configuration.
ExperimentRunner::ExperimentRunner(const ExperimentConfig& config)
    : config_(config) {}

ExperimentResult ExperimentRunner::Run() {
  // Run the selected method and attach diagnostics.
  ValidateConfig(config_);
  ExperimentInstance instance = BuildInstance(config_);

  ExperimentResult result;
  switch (config_.method) {
    case Method::Schur:
      result = RunSchurMethod(instance, config_);
      break;
    case Method::FullSystem:
      result = RunFullSystemMethod(instance, config_);
      break;
    case Method::Clique:
      result = RunCliqueMethod(instance, config_);
      break;
    case Method::Direct:
      result = RunDirectMethod(instance, config_);
      break;
    case Method::FjDynamics:
      result = RunFjDynamicsMethod(instance, config_);
      break;
    default:
      throw std::invalid_argument("Unknown experiment method");
  }
  result.user_graph_weight =
      config_.use_real_data ? MeanEdgeWeight(instance.user_graph)
                            : ResolveUserGraphWeight(config_);
  FillDiagnostics(instance, result);
  return result;
}

}  // namespace fj
