#pragma once

#include "fj/graph/bipartite_csr.hpp"
#include "fj/operators/auu_operator.hpp"
#include "fj/operators/linear_operator.hpp"
#include "fj/solver/inner_solver.hpp"

namespace fj {

class SchurComplementOperator : public LinearOperator {
 public:
  // Build the Schur-complement operator on the user block.
  SchurComplementOperator(const AuuOperator& Auu, const BipartiteCsr& W,
                          const InnerSolver& inner);

  // Return number of rows.
  Index rows() const override { return n_; }
  // Return number of cols.
  Index cols() const override { return n_; }

  // Apply the Schur-complement operator to x_u.
  void Apply(const Vector& x, Vector& y) const override;

 private:
  Index n_;
  const AuuOperator* Auu_;
  const BipartiteCsr* W_;
  const InnerSolver* inner_;
  mutable Vector tmp_g_in_;
  mutable Vector tmp_g_out_;
  mutable Vector tmp_u_;
};

}  // namespace fj
