#include "fj/graph/bipartite_reverse_index.hpp"

#include <stdexcept>

namespace fj {

BipartiteReverseIndex::BipartiteReverseIndex(
    const BipartiteCsr& bipartite)
    : n_users_(bipartite.num_users()),
      n_groups_(bipartite.num_groups()),
      row_ptr_(static_cast<size_t>(n_groups_ + 1), 0) {
  // Count memberships incident to every group.
  const auto& source_rows = bipartite.row_ptr();
  const auto& source_cols = bipartite.col_idx();
  const auto& source_values = bipartite.values();
  for (Index idx = 0; idx < bipartite.nnz(); ++idx) {
    const Index group = source_cols[static_cast<size_t>(idx)];
    if (group < 0 || group >= n_groups_) {
      throw std::invalid_argument("Bipartite group index out of range");
    }
    if (source_values[static_cast<size_t>(idx)] < 0.0) {
      throw std::invalid_argument("Bipartite weights must be nonnegative");
    }
    row_ptr_[static_cast<size_t>(group + 1)] += 1;
  }

  for (Index group = 0; group < n_groups_; ++group) {
    row_ptr_[static_cast<size_t>(group + 1)] +=
        row_ptr_[static_cast<size_t>(group)];
  }

  user_idx_.resize(static_cast<size_t>(bipartite.nnz()));
  values_.resize(static_cast<size_t>(bipartite.nnz()));
  std::vector<Index> offsets = row_ptr_;

  // Scatter each user-major membership into its group-major row.
  for (Index user = 0; user < n_users_; ++user) {
    const Index begin = source_rows[static_cast<size_t>(user)];
    const Index end = source_rows[static_cast<size_t>(user + 1)];
    for (Index idx = begin; idx < end; ++idx) {
      const Index group = source_cols[static_cast<size_t>(idx)];
      const Index pos = offsets[static_cast<size_t>(group)]++;
      user_idx_[static_cast<size_t>(pos)] = user;
      values_[static_cast<size_t>(pos)] =
          source_values[static_cast<size_t>(idx)];
    }
  }
}

}  // namespace fj

