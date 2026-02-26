#include "fj/operators/auu_operator.hpp"

#include <stdexcept>

namespace fj {

// Initialize the user-layer operator and cache cross-degree data.
AuuOperator::AuuOperator(const WeightedCsrGraph& user_graph,
                         const BipartiteCsr& bipartite,
                         const Vector& lambda_u)
    : n_(user_graph.num_nodes()),
      user_graph_(&user_graph),
      bipartite_(&bipartite),
      lambda_u_(lambda_u) {
  if (lambda_u_.size() != n_) {
    throw std::invalid_argument("lambda_u size mismatch");
  }
  if (bipartite_->num_users() != n_) {
    throw std::invalid_argument("bipartite user size mismatch");
  }
  cross_degree_ = bipartite_->user_degree();
  tmp_.setZero(n_);
}

void AuuOperator::Apply(const Vector& x, Vector& y) const {
  // Dispatch to the ref-based implementation to avoid copies.
  ApplyRef(x, y);
}

void AuuOperator::ApplyRef(const Eigen::Ref<const Vector>& x, Vector& y) const {
  // Compute (Lambda_u + D_u^c) x + L_u x.
  if (x.size() != n_) {
    throw std::invalid_argument("AuuOperator input size mismatch");
  }
  y = lambda_u_.cwiseProduct(x) + cross_degree_.cwiseProduct(x);
  if (user_graph_ && user_graph_->nnz() > 0) {
    user_graph_->laplacian_matvec(x, tmp_);
    y += tmp_;
  }
}

}  // namespace fj
