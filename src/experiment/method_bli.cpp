#include "fj/experiment/methods.hpp"

#include <utility>

#include "fj/baseline/bli_solver.hpp"
#include "fj/common/timer.hpp"
#include "fj/experiment/residuals.hpp"
#include "fj/metrics/metrics.hpp"

namespace fj {

ExperimentResult RunBliMethod(const ExperimentInstance& instance,
                              const ExperimentConfig& config) {
  // Run the KDD local residual-push method with unit relaxation.
  BliOptions options;
  options.max_rounds = config.outer_max_iters;
  options.max_updates = config.bli_max_updates;
  options.tolerance = config.outer_tol;
  options.omega = 1.0;
  options.record_history = !config.trace_csv_path.empty();

  Timer timer;
  BliResult solution = BliSolver::Solve(
      instance.user_graph, instance.group_graph, instance.bipartite,
      instance.lambda_u, instance.lambda_g, instance.b_u, instance.b_g,
      options);
  const double elapsed = timer.ElapsedSeconds();

  ExperimentResult result;
  result.method = config.method;
  result.tag = config.tag;
  result.outer_iters = solution.rounds;
  result.local_updates = solution.updates;
  result.seconds = elapsed;
  result.full_relative_residual = solution.full_relative_residual;
  result.relative_residual =
      SchurRelativeResidual(instance, config, solution.x_u);
  result.disagreement =
      GraphDisagreement(instance.user_graph, solution.x_u);
  result.internal_conflict = InternalConflict(solution.x_u, instance.s_u);
  result.polarization = Polarization(solution.x_u);
  result.controversy = Controversy(solution.x_u);
  for (const ResidualTracePoint& point : solution.residual_history) {
    result.convergence_trace.push_back(
        {"outer", 0, point.iteration, point.relative_residual, point.seconds});
  }
  result.x_u = std::move(solution.x_u);
  return result;
}

}  // namespace fj
