#include "fj/experiment/methods.hpp"

#include <limits>

#include "fj/baseline/pf_qe.hpp"
#include "fj/common/timer.hpp"

namespace fj {

ExperimentResult RunPfQeMethod(const ExperimentInstance& instance,
                               const ExperimentConfig& config) {
  // Estimate user-side quantities without constructing a full opinion vector.
  PfQeOptions options;
  options.sample_users = config.pf_sample_users;
  options.sample_edges = config.pf_sample_edges;
  options.forest_samples = config.pf_forest_samples;
  options.max_walk_steps = config.pf_max_walk_steps;
  options.seed = config.seed;

  Timer timer;
  PfQeResult estimate = PfQeEstimator::Estimate(
      instance.user_graph, instance.group_graph, instance.bipartite,
      instance.lambda_u, instance.lambda_g, instance.s_u, instance.s_g,
      options);
  const double elapsed = timer.ElapsedSeconds();

  ExperimentResult result;
  result.method = config.method;
  result.tag = config.tag;
  result.seconds = elapsed;
  result.relative_residual = std::numeric_limits<double>::quiet_NaN();
  result.disagreement = estimate.disagreement;
  result.internal_conflict = estimate.internal_conflict;
  result.polarization = estimate.polarization;
  result.controversy = estimate.controversy;
  result.sampled_users = estimate.sampled_users;
  result.sampled_edges = estimate.sampled_edges;
  result.forest_samples = estimate.forest_samples;
  result.walk_steps = estimate.walk_steps;
  return result;
}

}  // namespace fj

