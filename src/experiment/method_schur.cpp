#include "fj/experiment/methods.hpp"

#include <utility>

#include "fj/common/timer.hpp"
#include "fj/experiment/convergence_trace.hpp"
#include "fj/io/precond_io.hpp"
#include "fj/metrics/metrics.hpp"
#include "fj/operators/agg_operator.hpp"
#include "fj/operators/auu_operator.hpp"
#include "fj/operators/schur_operator.hpp"
#include "fj/preconditioner/jacobi_preconditioner.hpp"
#include "fj/preconditioner/precond_builder.hpp"
#include "fj/solver/cg_solver.hpp"
#include "fj/solver/inner_solver.hpp"

namespace fj {

ExperimentResult RunSchurMethod(const ExperimentInstance& instance,
                                const ExperimentConfig& config) {
  // Solve the user block via the Schur-complement formulation.
  AuuOperator Auu(instance.user_graph, instance.bipartite, instance.lambda_u);
  AggOperator Agg(instance.group_graph, instance.bipartite, instance.lambda_g);

  Vector agg_diag = BuildAggDiagonal(instance);

  const bool group_has_edges = instance.group_graph.nnz() > 0;
  const bool record_history = !config.trace_csv_path.empty();
  CgSolver inner_cg(config.inner_max_iters, config.inner_tol, record_history);
  JacobiPreconditioner jacobi(agg_diag);

  DiagonalInnerSolver diag_inner(agg_diag);
  IterativeInnerSolver iter_inner(Agg, inner_cg, &jacobi);

  // Select inner solver based on group-graph connectivity.
  const InnerSolver& inner = group_has_edges
                                 ? static_cast<const InnerSolver&>(iter_inner)
                                 : static_cast<const InnerSolver&>(diag_inner);

  SchurComplementOperator S(Auu, instance.bipartite, inner);

  // Eliminate the group block from both the matrix and the right-hand side:
  // S x_u = b_u + W Agg^{-1} b_g.
  Vector schur_rhs = instance.b_u;

  // Solve the outer system with CG.
  CgSolver outer_cg(config.outer_max_iters, config.outer_tol, record_history);
  Vector x_u;
  JacobiPreconditioner schur_jacobi;
  const Preconditioner* outer_precond = nullptr;
  if (config.schur_use_jacobi) {
    Vector diag;
    if (!config.schur_precond_path.empty()) {
      const PrecondMetadata metadata = PrecondIO::MakeMetadata(
          instance.user_graph, instance.group_graph, instance.bipartite,
          config.lambda_user, config.lambda_group, config.user_graph_scale,
          config.group_graph_scale);
      diag = PrecondIO::ReadJacobiDiag(
          config.schur_precond_path, PrecondKind::kSchurJacobi,
          instance.bipartite.num_users(), metadata);
    } else {
      diag = BuildSchurJacobiDiagonal(instance);
    }
    schur_jacobi.SetDiagonal(diag);
    outer_precond = &schur_jacobi;
  } else if (!config.schur_precond_path.empty()) {
    throw std::invalid_argument("schur_precond requires schur_jacobi=true");
  }
  inner.ResetStats();
  Timer timer;
  if (instance.b_g.squaredNorm() > 0.0) {
    Vector group_response;
    Vector user_correction;
    inner.Solve(instance.b_g, group_response);
    instance.bipartite.mul_W(group_response, user_correction);
    schur_rhs += user_correction;
  }
  SolverStats stats = outer_cg.Solve(S, schur_rhs, x_u, outer_precond);
  const double elapsed = timer.ElapsedSeconds();
  InnerSolveStats inner_stats = inner.Stats();

  ExperimentResult result;
  result.method = config.method;
  result.tag = config.tag;
  result.outer_iters = stats.iterations;
  result.inner_iters = inner_stats.iterations;
  result.inner_seconds = inner_stats.seconds;
  result.seconds = elapsed;
  result.relative_residual = stats.relative_residual;
  result.disagreement = GraphDisagreement(instance.user_graph, x_u);
  result.internal_conflict = InternalConflict(x_u, instance.s_u);
  result.polarization = Polarization(x_u);
  result.controversy = Controversy(x_u);
  AppendSolverTrace("outer", 0, stats, result);
  AppendInnerTrace(inner_stats, result);
  result.x_u = std::move(x_u);
  return result;
}

}  // namespace fj
