#pragma once

#include <vector>

#include "fj/common/types.hpp"

namespace fj {

class BipartiteCsr {
 public:
  // Construct an empty bipartite matrix.
  BipartiteCsr();
  // Construct an empty bipartite matrix with sizes.
  BipartiteCsr(Index n_users, Index n_groups);
  // Construct from CSR buffers.
  BipartiteCsr(Index n_users, Index n_groups, std::vector<Index> row_ptr,
               std::vector<Index> col_idx, std::vector<Scalar> values);

  // Build from triplets.
  static BipartiteCsr FromTriplets(Index n_users, Index n_groups,
                                   const std::vector<Triplet>& triplets);
  // Build from CSR buffers.
  static BipartiteCsr FromCsr(Index n_users, Index n_groups,
                              std::vector<Index> row_ptr,
                              std::vector<Index> col_idx,
                              std::vector<Scalar> values);

  // Return number of users.
  Index num_users() const { return n_users_; }
  // Return number of groups.
  Index num_groups() const { return n_groups_; }
  // Return number of nonzeros.
  Index nnz() const { return static_cast<Index>(values_.size()); }

  // Access raw CSR buffers.
  const std::vector<Index>& row_ptr() const { return row_ptr_; }
  const std::vector<Index>& col_idx() const { return col_idx_; }
  const std::vector<Scalar>& values() const { return values_; }
  std::vector<Scalar>& values_mut();

  // Compute y_u = W * x_g.
  void mul_W(const Eigen::Ref<const Vector>& x_g, Vector& y_u) const;
  // Compute y_g = W^T * x_u.
  void mul_Wt(const Eigen::Ref<const Vector>& x_u, Vector& y_g) const;

  // Return cached user cross-degrees.
  const Vector& user_degree() const;
  // Return cached group cross-degrees.
  const Vector& group_degree() const;

 private:
  void EnsureDegrees() const;

  Index n_users_;
  Index n_groups_;
  std::vector<Index> row_ptr_;
  std::vector<Index> col_idx_;
  std::vector<Scalar> values_;
  mutable Vector user_degree_;
  mutable Vector group_degree_;
  mutable bool degree_ready_;
};

}  // namespace fj
