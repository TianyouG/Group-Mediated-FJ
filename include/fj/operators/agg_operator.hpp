#pragma once

#include "fj/common/types.hpp"
#include "fj/graph/bipartite_csr.hpp"
#include "fj/graph/csr_graph.hpp"
#include "fj/operators/linear_operator.hpp"

namespace fj {

class AggOperator : public LinearOperator {
 public:
  // Build Agg = Lambda_g + L_g + D_g^c for the group layer.
  AggOperator(const WeightedCsrGraph& group_graph, const BipartiteCsr& bipartite,
              const Vector& lambda_g);

  // Return number of rows.
  Index rows() const override { return m_; }
  // Return number of cols.
  Index cols() const override { return m_; }

  // Apply operator to x (allocating output).
  void Apply(const Vector& x, Vector& y) const override;
  // Apply operator using a non-owning view into x.
  void ApplyRef(const Eigen::Ref<const Vector>& x, Vector& y) const;

  // Return group-layer lambda vector.
  const Vector& lambda_g() const { return lambda_g_; }
  // Return cross-layer degree vector.
  const Vector& cross_degree() const { return cross_degree_; }

 private:
  Index m_;
  const WeightedCsrGraph* group_graph_;
  const BipartiteCsr* bipartite_;
  Vector lambda_g_;
  Vector cross_degree_;
  mutable Vector tmp_;
};

}  // namespace fj
