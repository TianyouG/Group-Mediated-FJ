#pragma once

#include <vector>

#include "fj/common/types.hpp"
#include "fj/graph/bipartite_csr.hpp"

namespace fj {

// Group-major view of a user-major bipartite CSR matrix.
class BipartiteReverseIndex {
 public:
  // Build group-to-user adjacency from the existing user-to-group CSR data.
  explicit BipartiteReverseIndex(const BipartiteCsr& bipartite);

  // Return number of users.
  Index num_users() const { return n_users_; }
  // Return number of groups.
  Index num_groups() const { return n_groups_; }
  // Return number of memberships.
  Index nnz() const { return static_cast<Index>(values_.size()); }

  // Access group-major CSR buffers.
  const std::vector<Index>& row_ptr() const { return row_ptr_; }
  const std::vector<Index>& user_idx() const { return user_idx_; }
  const std::vector<Scalar>& values() const { return values_; }

 private:
  Index n_users_;
  Index n_groups_;
  std::vector<Index> row_ptr_;
  std::vector<Index> user_idx_;
  std::vector<Scalar> values_;
};

}  // namespace fj

