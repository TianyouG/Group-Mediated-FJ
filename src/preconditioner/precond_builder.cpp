#include "fj/preconditioner/precond_builder.hpp"

#include <algorithm>
#include <stdexcept>

namespace fj {

Vector BuildAggDiagonal(const ExperimentInstance& instance) {
  // Build diagonal for group-layer operator Agg.
  const Index m = instance.bipartite.num_groups();
  Vector agg_diag = instance.lambda_g + instance.bipartite.group_degree();
  if (instance.group_graph.nnz() > 0) {
    agg_diag += instance.group_graph.degree();
  }
  if (agg_diag.size() != m) {
    throw std::invalid_argument("Agg diagonal size mismatch");
  }
  return agg_diag;
}

Vector BuildFullJacobiDiagonal(const ExperimentInstance& instance) {
  // Build full-system Jacobi diagonal [diag_u; diag_g].
  const Index n_users = instance.bipartite.num_users();
  const Index n_groups = instance.bipartite.num_groups();

  Vector diag_u = instance.lambda_u + instance.bipartite.user_degree();
  if (instance.user_graph.nnz() > 0) {
    diag_u += instance.user_graph.degree();
  }

  Vector diag_g = BuildAggDiagonal(instance);

  Vector diag_full(n_users + n_groups);
  diag_full.head(n_users) = diag_u;
  diag_full.tail(n_groups) = diag_g;
  return diag_full;
}

Vector BuildSchurJacobiDiagonal(const ExperimentInstance& instance) {
  // Approximate the Schur complement diagonal with a Jacobi surrogate.
  const Index n = instance.bipartite.num_users();
  Vector diag = instance.lambda_u + instance.bipartite.user_degree();
  if (instance.user_graph.nnz() > 0) {
    diag += instance.user_graph.degree();
  }

  Vector agg_diag = BuildAggDiagonal(instance);

  Vector correction = Vector::Zero(n);
  const auto& row_ptr = instance.bipartite.row_ptr();
  const auto& col_idx = instance.bipartite.col_idx();
  const auto& values = instance.bipartite.values();
  for (Index u = 0; u < n; ++u) {
    const Index start = row_ptr[static_cast<size_t>(u)];
    const Index end = row_ptr[static_cast<size_t>(u + 1)];
    for (Index idx = start; idx < end; ++idx) {
      const Index g = col_idx[static_cast<size_t>(idx)];
      const double denom = agg_diag[g];
      if (denom > 0.0) {
        const double w = values[static_cast<size_t>(idx)];
        correction[u] += (w * w) / denom;
      }
    }
  }

  diag -= correction;

  const double kMinDiag = 1e-12;
  for (Index i = 0; i < diag.size(); ++i) {
    if (diag[i] <= kMinDiag) {
      diag[i] = kMinDiag;
    }
  }
  return diag;
}

}  // namespace fj
