#include "fj/experiment/methods.hpp"

#include <stdexcept>

#include "fj/baseline/direct_solver.hpp"
#include "fj/common/timer.hpp"
#include "fj/experiment/residuals.hpp"
#include "fj/metrics/metrics.hpp"
#include "fj/operators/agg_operator.hpp"
#include "fj/operators/auu_operator.hpp"
#include "fj/operators/full_system_operator.hpp"

namespace fj {

ExperimentResult RunDirectMethod(const ExperimentInstance& instance,
                                 const ExperimentConfig& config) {
  // Solve the full system via dense direct factorization.
  const Index n_users = instance.bipartite.num_users();
  const Index n_groups = instance.bipartite.num_groups();
  const Index n_total = n_users + n_groups;

  if (n_total > config.direct_max_dim) {
    throw std::runtime_error("Direct method exceeds direct_max_dim");
  }

  AuuOperator Auu(instance.user_graph, instance.bipartite, instance.lambda_u);
  AggOperator Agg(instance.group_graph, instance.bipartite, instance.lambda_g);
  FullSystemOperator full(Auu, Agg, instance.bipartite);

  Vector b_full(n_total);
  b_full.head(n_users) = instance.b_u;
  b_full.tail(n_groups) = instance.b_g;

  // Time the direct solve.
  Timer timer;
  Vector x_full = DirectSolver::Solve(full, b_full);
  const double elapsed = timer.ElapsedSeconds();

  Vector x_u = x_full.head(n_users);

  ExperimentResult result;
  result.method = config.method;
  result.tag = config.tag;
  result.outer_iters = 0;
  result.inner_iters = 0;
  result.seconds = elapsed;
  result.relative_residual = SchurRelativeResidual(instance, config, x_u);
  result.disagreement = GraphDisagreement(instance.user_graph, x_u);
  result.internal_conflict = InternalConflict(x_u, instance.s_u);
  result.polarization = Polarization(x_u);
  result.controversy = Controversy(x_u);
  return result;
}

}  // namespace fj
