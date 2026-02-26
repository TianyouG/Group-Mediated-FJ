#pragma once

#include "fj/graph/bipartite_csr.hpp"
#include "fj/operators/agg_operator.hpp"
#include "fj/operators/auu_operator.hpp"
#include "fj/operators/linear_operator.hpp"

namespace fj {

class FullSystemOperator : public LinearOperator {
 public:
  // Build the full block operator from Auu, Agg, and W.
  FullSystemOperator(const AuuOperator& Auu, const AggOperator& Agg,
                     const BipartiteCsr& W);

  // Return number of rows.
  Index rows() const override { return n_ + m_; }
  // Return number of cols.
  Index cols() const override { return n_ + m_; }

  // Apply the full block operator to [x_u; x_g].
  void Apply(const Vector& x, Vector& y) const override;

 private:
  Index n_;
  Index m_;
  const AuuOperator* Auu_;
  const AggOperator* Agg_;
  const BipartiteCsr* W_;
  mutable Vector tmp_u_;
  mutable Vector tmp_g_;
};

}  // namespace fj
