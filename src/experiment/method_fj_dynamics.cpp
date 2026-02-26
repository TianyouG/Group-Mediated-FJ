#include "fj/experiment/methods.hpp"

#include <cmath>

#include "fj/common/timer.hpp"
#include "fj/experiment/residuals.hpp"
#include "fj/metrics/metrics.hpp"

namespace fj {
namespace {

void ComputeAdjacencyProduct(const ExperimentInstance& instance,
                             const Vector& x_u, const Vector& x_g,
                             Vector& ax_u, Vector& ax_g, Vector& tmp_u,
                             Vector& tmp_g) {
  // Compute the full-graph adjacency product A * x.
  if (instance.user_graph.nnz() > 0) {
    instance.user_graph.matvec(x_u, ax_u);
  } else {
    ax_u.setZero(x_u.size());
  }
  if (instance.group_graph.nnz() > 0) {
    instance.group_graph.matvec(x_g, ax_g);
  } else {
    ax_g.setZero(x_g.size());
  }

  instance.bipartite.mul_W(x_g, tmp_u);
  ax_u += tmp_u;
  instance.bipartite.mul_Wt(x_u, tmp_g);
  ax_g += tmp_g;
}

double ComputeRelativeResidual(const Vector& denom_u, const Vector& denom_g,
                               const Vector& b_u, const Vector& b_g,
                               const Vector& x_u, const Vector& x_g,
                               const ExperimentInstance& instance,
                               Vector& ax_u, Vector& ax_g, Vector& tmp_u,
                               Vector& tmp_g, Vector& r_u, Vector& r_g,
                               double b_norm) {
  // Compute relative residual of (Lambda + L) x = b using cached buffers.
  ComputeAdjacencyProduct(instance, x_u, x_g, ax_u, ax_g, tmp_u, tmp_g);
  r_u = denom_u.cwiseProduct(x_u) - ax_u - b_u;
  r_g = denom_g.cwiseProduct(x_g) - ax_g - b_g;
  const double res_norm =
      std::sqrt(r_u.squaredNorm() + r_g.squaredNorm());
  return res_norm / b_norm;
}

}  // namespace

ExperimentResult RunFjDynamicsMethod(const ExperimentInstance& instance,
                                     const ExperimentConfig& config) {
  // Run the fixed-point FJ dynamics iteration on the full graph.
  const Index n_users = instance.bipartite.num_users();
  const Index n_groups = instance.bipartite.num_groups();

  Vector degree_u = instance.bipartite.user_degree();
  if (instance.user_graph.nnz() > 0) {
    degree_u += instance.user_graph.degree();
  }
  Vector degree_g = instance.bipartite.group_degree();
  if (instance.group_graph.nnz() > 0) {
    degree_g += instance.group_graph.degree();
  }

  Vector denom_u = degree_u + instance.lambda_u;
  Vector denom_g = degree_g + instance.lambda_g;

  Vector inv_denom_u(n_users);
  for (Index i = 0; i < n_users; ++i) {
    inv_denom_u[i] = denom_u[i] > 0.0 ? 1.0 / denom_u[i] : 0.0;
  }
  Vector inv_denom_g(n_groups);
  for (Index i = 0; i < n_groups; ++i) {
    inv_denom_g[i] = denom_g[i] > 0.0 ? 1.0 / denom_g[i] : 0.0;
  }

  Vector x_u = Vector::Zero(n_users);
  Vector x_g = Vector::Zero(n_groups);
  Vector x_u_new(n_users);
  Vector x_g_new(n_groups);

  Vector ax_u(n_users);
  Vector ax_g(n_groups);
  Vector tmp_u(n_users);
  Vector tmp_g(n_groups);
  Vector r_u(n_users);
  Vector r_g(n_groups);

  double b_norm =
      std::sqrt(instance.b_u.squaredNorm() + instance.b_g.squaredNorm());
  if (b_norm == 0.0) {
    b_norm = 1.0;
  }

  Timer timer;
  Index iterations = 0;
  double rel_res = 0.0;
  bool converged = false;

  for (Index iter = 0; iter < config.outer_max_iters; ++iter) {
    rel_res = ComputeRelativeResidual(denom_u, denom_g, instance.b_u,
                                      instance.b_g, x_u, x_g, instance, ax_u,
                                      ax_g, tmp_u, tmp_g, r_u, r_g, b_norm);
    if (rel_res <= config.outer_tol) {
      converged = true;
      break;
    }

    x_u_new = (ax_u + instance.b_u).cwiseProduct(inv_denom_u);
    x_g_new = (ax_g + instance.b_g).cwiseProduct(inv_denom_g);
    x_u.swap(x_u_new);
    x_g.swap(x_g_new);
    iterations += 1;
  }

  if (!converged) {
    rel_res = ComputeRelativeResidual(denom_u, denom_g, instance.b_u,
                                      instance.b_g, x_u, x_g, instance, ax_u,
                                      ax_g, tmp_u, tmp_g, r_u, r_g, b_norm);
  }

  const double elapsed = timer.ElapsedSeconds();

  ExperimentResult result;
  result.method = config.method;
  result.tag = config.tag;
  result.outer_iters = iterations;
  result.inner_iters = 0;
  result.inner_seconds = 0.0;
  result.seconds = elapsed;
  result.relative_residual = SchurRelativeResidual(instance, config, x_u);
  result.disagreement = GraphDisagreement(instance.user_graph, x_u);
  result.internal_conflict = InternalConflict(x_u, instance.s_u);
  result.polarization = Polarization(x_u);
  result.controversy = Controversy(x_u);
  return result;
}

}  // namespace fj
