#include "fj/operators/agg_operator.hpp"

#include <stdexcept>

namespace fj {

// Initialize the group-layer operator and cache cross-degree data.
AggOperator::AggOperator(const WeightedCsrGraph& group_graph,
                         const BipartiteCsr& bipartite,
                         const Vector& lambda_g)
    : m_(group_graph.num_nodes()),
      group_graph_(&group_graph),
      bipartite_(&bipartite),
      lambda_g_(lambda_g) {
  if (lambda_g_.size() != m_) {
    throw std::invalid_argument("lambda_g size mismatch");
  }
  if (bipartite_->num_groups() != m_) {
    throw std::invalid_argument("bipartite group size mismatch");
  }
  cross_degree_ = bipartite_->group_degree();
  tmp_.setZero(m_);
}

void AggOperator::Apply(const Vector& x, Vector& y) const {
  // Dispatch to the ref-based implementation to avoid copies.
  ApplyRef(x, y);
}

void AggOperator::ApplyRef(const Eigen::Ref<const Vector>& x, Vector& y) const {
  // Compute (Lambda_g + D_g^c) x + L_g x.
  if (x.size() != m_) {
    throw std::invalid_argument("AggOperator input size mismatch");
  }
  y = lambda_g_.cwiseProduct(x) + cross_degree_.cwiseProduct(x);
  if (group_graph_ && group_graph_->nnz() > 0) {
    group_graph_->laplacian_matvec(x, tmp_);
    y += tmp_;
  }
}

}  // namespace fj
