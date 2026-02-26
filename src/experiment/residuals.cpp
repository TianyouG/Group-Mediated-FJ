#include "fj/experiment/residuals.hpp"

#include <stdexcept>

#include "fj/operators/agg_operator.hpp"
#include "fj/operators/auu_operator.hpp"
#include "fj/operators/schur_operator.hpp"
#include "fj/preconditioner/jacobi_preconditioner.hpp"
#include "fj/solver/cg_solver.hpp"
#include "fj/solver/inner_solver.hpp"

namespace fj {

double SchurRelativeResidual(const ExperimentInstance& instance,
                             const ExperimentConfig& config,
                             const Vector& x_u) {
  const Index n_users = instance.bipartite.num_users();
  if (x_u.size() != n_users) {
    throw std::invalid_argument("SchurRelativeResidual size mismatch");
  }

  AuuOperator Auu(instance.user_graph, instance.bipartite, instance.lambda_u);
  AggOperator Agg(instance.group_graph, instance.bipartite, instance.lambda_g);

  Vector agg_diag = instance.lambda_g + instance.bipartite.group_degree();
  if (instance.group_graph.nnz() > 0) {
    agg_diag += instance.group_graph.degree();
  }

  const bool group_has_edges = instance.group_graph.nnz() > 0;
  CgSolver inner_cg(config.inner_max_iters, config.inner_tol);
  JacobiPreconditioner jacobi(agg_diag);
  DiagonalInnerSolver diag_inner(agg_diag);
  IterativeInnerSolver iter_inner(Agg, inner_cg, &jacobi);

  const InnerSolver& inner = group_has_edges
                                 ? static_cast<const InnerSolver&>(iter_inner)
                                 : static_cast<const InnerSolver&>(diag_inner);

  SchurComplementOperator S(Auu, instance.bipartite, inner);

  Vector Sx(n_users);
  S.Apply(x_u, Sx);
  Vector r = instance.b_u - Sx;
  double denom = instance.b_u.norm();
  if (denom == 0.0) {
    denom = 1.0;
  }
  return r.norm() / denom;
}

}  // namespace fj
