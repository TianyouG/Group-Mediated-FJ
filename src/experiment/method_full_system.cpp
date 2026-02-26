#include "fj/experiment/methods.hpp"

#include "fj/common/timer.hpp"
#include "fj/experiment/residuals.hpp"
#include "fj/metrics/metrics.hpp"
#include "fj/operators/agg_operator.hpp"
#include "fj/operators/auu_operator.hpp"
#include "fj/operators/full_system_operator.hpp"
#include "fj/io/precond_io.hpp"
#include "fj/preconditioner/jacobi_preconditioner.hpp"
#include "fj/preconditioner/precond_builder.hpp"
#include "fj/solver/cg_solver.hpp"

namespace fj {

ExperimentResult RunFullSystemMethod(const ExperimentInstance& instance,
                                     const ExperimentConfig& config) {
  // Solve the full block system with CG/PCG.
  const Index n_users = instance.bipartite.num_users();
  const Index n_groups = instance.bipartite.num_groups();

  AuuOperator Auu(instance.user_graph, instance.bipartite, instance.lambda_u);
  AggOperator Agg(instance.group_graph, instance.bipartite, instance.lambda_g);
  FullSystemOperator full(Auu, Agg, instance.bipartite);

  Vector b_full(n_users + n_groups);
  b_full.head(n_users) = instance.b_u;
  b_full.tail(n_groups) = instance.b_g;

  // Optionally use Jacobi preconditioning on the full system.
  JacobiPreconditioner jacobi;
  const Preconditioner* precond = nullptr;
  if (config.full_system_use_jacobi) {
    Vector diag_full;
    if (!config.full_precond_path.empty()) {
      diag_full = PrecondIO::ReadJacobiDiag(
          config.full_precond_path, PrecondKind::kFullJacobi,
          n_users + n_groups, n_users, n_groups, config.lambda_user,
          config.lambda_group);
    } else {
      diag_full = BuildFullJacobiDiagonal(instance);
    }
    jacobi.SetDiagonal(diag_full);
    precond = &jacobi;
  } else if (!config.full_precond_path.empty()) {
    throw std::invalid_argument("full_precond requires full_system_jacobi=true");
  }

  // Run CG on the full operator.
  CgSolver cg(config.outer_max_iters, config.outer_tol);
  Vector x_full;
  Timer timer;
  SolverStats stats = cg.Solve(full, b_full, x_full, precond);
  const double elapsed = timer.ElapsedSeconds();

  Vector x_u = x_full.head(n_users);

  ExperimentResult result;
  result.method = config.method;
  result.tag = config.tag;
  result.outer_iters = stats.iterations;
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
