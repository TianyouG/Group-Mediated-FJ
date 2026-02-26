#pragma once

#include <vector>

#include "fj/common/types.hpp"

namespace fj {

class WeightedCsrGraph {
 public:
  // Construct an empty graph.
  WeightedCsrGraph();
  // Construct an empty graph with n nodes.
  explicit WeightedCsrGraph(Index n);
  // Construct a graph from CSR buffers.
  WeightedCsrGraph(Index n, std::vector<Index> row_ptr,
                   std::vector<Index> col_idx,
                   std::vector<Scalar> values);

  // Build a graph from triplets; optionally symmetrize.
  static WeightedCsrGraph FromTriplets(Index n,
                                       const std::vector<Triplet>& triplets,
                                       bool symmetrize = false);

  // Build a graph from CSR buffers.
  static WeightedCsrGraph FromCsr(Index n, std::vector<Index> row_ptr,
                                  std::vector<Index> col_idx,
                                  std::vector<Scalar> values);

  // Return number of nodes.
  Index num_nodes() const { return n_; }
  // Return number of non-zero entries in adjacency.
  Index nnz() const { return static_cast<Index>(values_.size()); }

  // Access raw CSR buffers.
  const std::vector<Index>& row_ptr() const { return row_ptr_; }
  const std::vector<Index>& col_idx() const { return col_idx_; }
  const std::vector<Scalar>& values() const { return values_; }

  // Sum of all stored edge weights.
  double total_weight() const;

  // Return degree vector (computed lazily).
  const Vector& degree() const;

  // Compute y = A x.
  void matvec(const Eigen::Ref<const Vector>& x, Vector& y) const;
  // Compute y = L x where L = D - A.
  void laplacian_matvec(const Eigen::Ref<const Vector>& x, Vector& y) const;

 private:
  void EnsureDegree() const;

  Index n_;
  std::vector<Index> row_ptr_;
  std::vector<Index> col_idx_;
  std::vector<Scalar> values_;
  mutable Vector degree_;
  mutable bool degree_ready_;
};

}  // namespace fj
