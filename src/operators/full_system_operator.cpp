#include "fj/operators/full_system_operator.hpp"

#include <stdexcept>

namespace fj {

// Initialize the full system operator with cached scratch buffers.
FullSystemOperator::FullSystemOperator(const AuuOperator& Auu,
                                       const AggOperator& Agg,
                                       const BipartiteCsr& W)
    : n_(Auu.rows()), m_(Agg.rows()), Auu_(&Auu), Agg_(&Agg), W_(&W) {
  if (W_->num_users() != n_ || W_->num_groups() != m_) {
    throw std::invalid_argument("FullSystemOperator size mismatch");
  }
  tmp_u_.setZero(n_);
  tmp_g_.setZero(m_);
}

void FullSystemOperator::Apply(const Vector& x, Vector& y) const {
  // Apply block operator: [Auu -W; -W^T Agg].
  if (x.size() != n_ + m_) {
    throw std::invalid_argument("FullSystemOperator input size mismatch");
  }
  y.resize(n_ + m_);

  const Eigen::Ref<const Vector> x_u(x.head(n_));
  const Eigen::Ref<const Vector> x_g(x.tail(m_));

  Auu_->ApplyRef(x_u, tmp_u_);
  W_->mul_W(x_g, tmp_g_);
  y.head(n_) = tmp_u_ - tmp_g_;

  Agg_->ApplyRef(x_g, tmp_g_);
  W_->mul_Wt(x_u, tmp_u_);
  y.tail(m_) = tmp_g_ - tmp_u_;
}

}  // namespace fj
