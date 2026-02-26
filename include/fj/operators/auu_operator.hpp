#pragma once

#include "fj/common/types.hpp"
#include "fj/graph/bipartite_csr.hpp"
#include "fj/graph/csr_graph.hpp"
#include "fj/operators/linear_operator.hpp"

namespace fj {

class AuuOperator : public LinearOperator {
 public:
  // Build Auu = Lambda_u + L_u + D_u^c for the user layer.
  AuuOperator(const WeightedCsrGraph& user_graph, const BipartiteCsr& bipartite,
              const Vector& lambda_u);

  // Return number of rows.
  Index rows() const override { return n_; }
  // Return number of cols.
  Index cols() const override { return n_; }

  // Apply operator to x (allocating output).
  void Apply(const Vector& x, Vector& y) const override;
  // Apply operator using a non-owning view into x.
  void ApplyRef(const Eigen::Ref<const Vector>& x, Vector& y) const;

  // Return user-layer lambda vector.
  const Vector& lambda_u() const { return lambda_u_; }
  // Return cross-layer degree vector.
  const Vector& cross_degree() const { return cross_degree_; }

 private:
  Index n_;
  const WeightedCsrGraph* user_graph_;
  const BipartiteCsr* bipartite_;
  Vector lambda_u_;
  Vector cross_degree_;
  mutable Vector tmp_;
};

}  // namespace fj
