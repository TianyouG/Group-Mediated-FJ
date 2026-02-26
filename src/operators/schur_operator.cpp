#include "fj/operators/schur_operator.hpp"

#include <stdexcept>

namespace fj {

// Initialize Schur operator with scratch buffers.
SchurComplementOperator::SchurComplementOperator(const AuuOperator& Auu,
                                                 const BipartiteCsr& W,
                                                 const InnerSolver& inner)
    : n_(Auu.rows()), Auu_(&Auu), W_(&W), inner_(&inner) {
  if (W_->num_users() != n_) {
    throw std::invalid_argument("Schur operator user size mismatch");
  }
  if (inner_->size() != W_->num_groups()) {
    throw std::invalid_argument("Schur operator inner size mismatch");
  }
  tmp_g_in_.setZero(W_->num_groups());
  tmp_g_out_.setZero(W_->num_groups());
  tmp_u_.setZero(n_);
}

void SchurComplementOperator::Apply(const Vector& x, Vector& y) const {
  // Compute Sx = Auu x - W * (Agg^{-1} (W^T x)).
  if (x.size() != n_) {
    throw std::invalid_argument("Schur operator input size mismatch");
  }

  W_->mul_Wt(x, tmp_g_in_);
  inner_->Solve(tmp_g_in_, tmp_g_out_);
  W_->mul_W(tmp_g_out_, tmp_u_);

  Auu_->Apply(x, y);
  y -= tmp_u_;
}

}  // namespace fj
