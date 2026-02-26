#include "fj/graph/bipartite_csr.hpp"

#include <stdexcept>

namespace fj {

// Initialize an empty bipartite matrix.
BipartiteCsr::BipartiteCsr() : n_users_(0), n_groups_(0), degree_ready_(false) {}

// Initialize an empty bipartite matrix with sizes.
BipartiteCsr::BipartiteCsr(Index n_users, Index n_groups)
    : n_users_(n_users),
      n_groups_(n_groups),
      row_ptr_(static_cast<size_t>(n_users + 1), 0),
      degree_ready_(false) {}

BipartiteCsr::BipartiteCsr(Index n_users, Index n_groups,
                           std::vector<Index> row_ptr,
                           std::vector<Index> col_idx,
                           std::vector<Scalar> values)
    : n_users_(n_users),
      n_groups_(n_groups),
      row_ptr_(std::move(row_ptr)),
      col_idx_(std::move(col_idx)),
      values_(std::move(values)),
      degree_ready_(false) {
  if (n_users_ < 0 || n_groups_ < 0) {
    throw std::invalid_argument("bipartite sizes must be nonnegative");
  }
  if (row_ptr_.size() != static_cast<size_t>(n_users_ + 1)) {
    throw std::invalid_argument("row_ptr size mismatch");
  }
  if (col_idx_.size() != values_.size()) {
    throw std::invalid_argument("CSR col/value size mismatch");
  }
  if (!row_ptr_.empty() &&
      static_cast<size_t>(row_ptr_.back()) != values_.size()) {
    throw std::invalid_argument("row_ptr tail mismatch with nnz");
  }
}

BipartiteCsr BipartiteCsr::FromTriplets(Index n_users, Index n_groups,
                                        const std::vector<Triplet>& triplets) {
  // Build CSR buffers from triplets.
  if (n_users <= 0 || n_groups <= 0) {
    return BipartiteCsr();
  }
  std::vector<Index> row_counts(static_cast<size_t>(n_users), 0);
  for (const auto& t : triplets) {
    const Index u = static_cast<Index>(t.row());
    const Index g = static_cast<Index>(t.col());
    if (u < 0 || g < 0 || u >= n_users || g >= n_groups) {
      throw std::invalid_argument("Triplet index out of range");
    }
    row_counts[static_cast<size_t>(u)] += 1;
  }

  std::vector<Index> row_ptr(static_cast<size_t>(n_users + 1), 0);
  for (Index u = 0; u < n_users; ++u) {
    row_ptr[static_cast<size_t>(u + 1)] =
        row_ptr[static_cast<size_t>(u)] + row_counts[static_cast<size_t>(u)];
  }

  const Index nnz = row_ptr.back();
  std::vector<Index> col_idx(static_cast<size_t>(nnz));
  std::vector<Scalar> values(static_cast<size_t>(nnz));
  std::vector<Index> offsets = row_ptr;

  for (const auto& t : triplets) {
    const Index u = static_cast<Index>(t.row());
    const Index g = static_cast<Index>(t.col());
    const Scalar w = static_cast<Scalar>(t.value());
    const Index pos = offsets[static_cast<size_t>(u)]++;
    col_idx[static_cast<size_t>(pos)] = g;
    values[static_cast<size_t>(pos)] = w;
  }

  return BipartiteCsr(n_users, n_groups, std::move(row_ptr),
                      std::move(col_idx), std::move(values));
}

BipartiteCsr BipartiteCsr::FromCsr(Index n_users, Index n_groups,
                                   std::vector<Index> row_ptr,
                                   std::vector<Index> col_idx,
                                   std::vector<Scalar> values) {
  return BipartiteCsr(n_users, n_groups, std::move(row_ptr),
                      std::move(col_idx), std::move(values));
}

void BipartiteCsr::mul_W(const Eigen::Ref<const Vector>& x_g,
                         Vector& y_u) const {
  // Multiply W by a group vector.
  if (x_g.size() != n_groups_) {
    throw std::invalid_argument("mul_W input size mismatch");
  }
  y_u.setZero(n_users_);
  for (Index u = 0; u < n_users_; ++u) {
    const Index start = row_ptr_[static_cast<size_t>(u)];
    const Index end = row_ptr_[static_cast<size_t>(u + 1)];
    Scalar sum = 0.0;
    for (Index idx = start; idx < end; ++idx) {
      sum += values_[static_cast<size_t>(idx)] *
             x_g[col_idx_[static_cast<size_t>(idx)]];
    }
    y_u[u] = sum;
  }
}

void BipartiteCsr::mul_Wt(const Eigen::Ref<const Vector>& x_u,
                          Vector& y_g) const {
  // Multiply W^T by a user vector.
  if (x_u.size() != n_users_) {
    throw std::invalid_argument("mul_Wt input size mismatch");
  }
  y_g.setZero(n_groups_);
  for (Index u = 0; u < n_users_; ++u) {
    const Scalar xu = x_u[u];
    const Index start = row_ptr_[static_cast<size_t>(u)];
    const Index end = row_ptr_[static_cast<size_t>(u + 1)];
    for (Index idx = start; idx < end; ++idx) {
      y_g[col_idx_[static_cast<size_t>(idx)]] +=
          values_[static_cast<size_t>(idx)] * xu;
    }
  }
}

const Vector& BipartiteCsr::user_degree() const {
  // Return cached user cross-degrees.
  EnsureDegrees();
  return user_degree_;
}

const Vector& BipartiteCsr::group_degree() const {
  // Return cached group cross-degrees.
  EnsureDegrees();
  return group_degree_;
}

void BipartiteCsr::EnsureDegrees() const {
  // Compute cross-degrees if not cached.
  if (degree_ready_) {
    return;
  }
  user_degree_.setZero(n_users_);
  group_degree_.setZero(n_groups_);
  for (Index u = 0; u < n_users_; ++u) {
    const Index start = row_ptr_[static_cast<size_t>(u)];
    const Index end = row_ptr_[static_cast<size_t>(u + 1)];
    Scalar sum = 0.0;
    for (Index idx = start; idx < end; ++idx) {
      const Index g = col_idx_[static_cast<size_t>(idx)];
      const Scalar w = values_[static_cast<size_t>(idx)];
      sum += w;
      group_degree_[g] += w;
    }
    user_degree_[u] = sum;
  }
  degree_ready_ = true;
}

std::vector<Scalar>& BipartiteCsr::values_mut() {
  degree_ready_ = false;
  return values_;
}

}  // namespace fj
