#include "fj/experiment/methods.hpp"

#include <stdexcept>
#include <vector>

#include "fj/common/timer.hpp"
#include "fj/metrics/metrics.hpp"
#include "fj/operators/linear_operator.hpp"
#include "fj/solver/cg_solver.hpp"

namespace fj {
namespace {

class ImplicitCliqueOperator : public LinearOperator {
 public:
  // Build an operator for Lambda_u + L_user + L_clique without explicit clique edges.
  ImplicitCliqueOperator(const WeightedCsrGraph& user_graph,
                         const BipartiteCsr& bipartite,
                         const Vector& lambda,
                         bool use_weights)
      : n_(user_graph.num_nodes()),
        m_(bipartite.num_groups()),
        user_graph_(&user_graph),
        bipartite_(&bipartite),
        lambda_(lambda),
        use_weights_(use_weights),
        user_degree_(nullptr),
        group_degree_(nullptr),
        tmp_u_(n_),
        tmp_g_(m_) {
    if (lambda_.size() != n_) {
      throw std::invalid_argument("ImplicitCliqueOperator lambda size mismatch");
    }
    if (bipartite_->num_users() != n_) {
      throw std::invalid_argument("ImplicitCliqueOperator user size mismatch");
    }
    if (m_ < 0) {
      throw std::invalid_argument("ImplicitCliqueOperator group size mismatch");
    }

    if (use_weights_) {
      user_degree_ = &bipartite_->user_degree();
      group_degree_ = &bipartite_->group_degree();
    } else {
      group_size_.assign(static_cast<size_t>(m_), 0);
      const auto& row_ptr = bipartite_->row_ptr();
      const auto& col_idx = bipartite_->col_idx();
      for (Index u = 0; u < n_; ++u) {
        const Index start = row_ptr[static_cast<size_t>(u)];
        const Index end = row_ptr[static_cast<size_t>(u + 1)];
        for (Index idx = start; idx < end; ++idx) {
          const Index g = col_idx[static_cast<size_t>(idx)];
          group_size_[static_cast<size_t>(g)] += 1;
        }
      }
      sum_group_sizes_.setZero(n_);
      for (Index u = 0; u < n_; ++u) {
        const Index start = row_ptr[static_cast<size_t>(u)];
        const Index end = row_ptr[static_cast<size_t>(u + 1)];
        double total = 0.0;
        for (Index idx = start; idx < end; ++idx) {
          const Index g = col_idx[static_cast<size_t>(idx)];
          total += static_cast<double>(group_size_[static_cast<size_t>(g)]);
        }
        sum_group_sizes_[u] = total;
      }
    }
  }

  // Return number of rows.
  Index rows() const override { return n_; }
  // Return number of cols.
  Index cols() const override { return n_; }

  void Apply(const Vector& x, Vector& y) const override {
    // Apply Lambda_u + L_user + implicit L_clique.
    if (x.size() != n_) {
      throw std::invalid_argument("ImplicitCliqueOperator input size mismatch");
    }
    y = lambda_.cwiseProduct(x);
    if (user_graph_ && user_graph_->nnz() > 0) {
      user_graph_->laplacian_matvec(x, tmp_u_);
      y += tmp_u_;
    }

    if (m_ == 0) {
      return;
    }

    if (use_weights_) {
      // L_clique x = D_u x - W D_g^{-1} W^T x (with weights).
      bipartite_->mul_Wt(x, tmp_g_);
      for (Index g = 0; g < m_; ++g) {
        const double denom = (*group_degree_)[g];
        tmp_g_[g] = denom > 0.0 ? (tmp_g_[g] / denom) : 0.0;
      }
      bipartite_->mul_W(tmp_g_, tmp_u_);
      y += user_degree_->cwiseProduct(x);
      y -= tmp_u_;
    } else {
      // Unweighted clique: treat every membership as weight 1.
      tmp_g_.setZero(m_);
      const auto& row_ptr = bipartite_->row_ptr();
      const auto& col_idx = bipartite_->col_idx();
      for (Index u = 0; u < n_; ++u) {
        const Index start = row_ptr[static_cast<size_t>(u)];
        const Index end = row_ptr[static_cast<size_t>(u + 1)];
        const double xu = x[u];
        for (Index idx = start; idx < end; ++idx) {
          const Index g = col_idx[static_cast<size_t>(idx)];
          tmp_g_[g] += xu;
        }
      }

      for (Index u = 0; u < n_; ++u) {
        tmp_u_[u] = sum_group_sizes_[u] * x[u];
      }
      for (Index u = 0; u < n_; ++u) {
        const Index start = row_ptr[static_cast<size_t>(u)];
        const Index end = row_ptr[static_cast<size_t>(u + 1)];
        for (Index idx = start; idx < end; ++idx) {
          const Index g = col_idx[static_cast<size_t>(idx)];
          tmp_u_[u] -= tmp_g_[g];
        }
      }
      y += tmp_u_;
    }
  }

 private:
  Index n_;
  Index m_;
  const WeightedCsrGraph* user_graph_;
  const BipartiteCsr* bipartite_;
  Vector lambda_;
  bool use_weights_;
  const Vector* user_degree_;
  const Vector* group_degree_;
  std::vector<Index> group_size_;
  Vector sum_group_sizes_;
  mutable Vector tmp_u_;
  mutable Vector tmp_g_;
};

}  // namespace

ExperimentResult RunCliqueMethod(const ExperimentInstance& instance,
                                 const ExperimentConfig& config) {
  // Solve on the implicit clique-expanded user graph.
  ImplicitCliqueOperator A(instance.user_graph, instance.bipartite,
                           instance.lambda_u, config.clique_use_weights);

  // Run CG on the user-only system.
  CgSolver cg(config.outer_max_iters, config.outer_tol);
  Vector x_u;
  Timer timer;
  SolverStats stats = cg.Solve(A, instance.b_u, x_u, nullptr);
  const double elapsed = timer.ElapsedSeconds();

  ExperimentResult result;
  result.method = config.method;
  result.tag = config.tag;
  result.outer_iters = stats.iterations;
  result.inner_iters = 0;
  result.seconds = elapsed;
  result.relative_residual = stats.relative_residual;
  result.disagreement = GraphDisagreement(instance.user_graph, x_u);
  result.internal_conflict = InternalConflict(x_u, instance.s_u);
  result.polarization = Polarization(x_u);
  result.controversy = Controversy(x_u);
  return result;
}

}  // namespace fj
