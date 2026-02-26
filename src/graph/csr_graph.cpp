#include "fj/graph/csr_graph.hpp"

#include <numeric>
#include <stdexcept>

namespace fj {

// Initialize an empty graph.
WeightedCsrGraph::WeightedCsrGraph() : n_(0), degree_ready_(false) {}

// Initialize an empty graph with n nodes.
WeightedCsrGraph::WeightedCsrGraph(Index n)
    : n_(n), row_ptr_(static_cast<size_t>(n + 1), 0), degree_ready_(false) {}

WeightedCsrGraph::WeightedCsrGraph(Index n, std::vector<Index> row_ptr,
                                   std::vector<Index> col_idx,
                                   std::vector<Scalar> values)
    : n_(n),
      row_ptr_(std::move(row_ptr)),
      col_idx_(std::move(col_idx)),
      values_(std::move(values)),
      degree_ready_(false) {
  if (n_ < 0) {
    throw std::invalid_argument("graph node count must be nonnegative");
  }
  if (row_ptr_.size() != static_cast<size_t>(n_ + 1)) {
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

WeightedCsrGraph WeightedCsrGraph::FromTriplets(
    Index n, const std::vector<Triplet>& triplets, bool symmetrize) {
  // Build CSR buffers from triplets with optional symmetrization.
  if (n <= 0) {
    return WeightedCsrGraph(0);
  }

  std::vector<Index> row_counts(static_cast<size_t>(n), 0);
  for (const auto& t : triplets) {
    const Index u = static_cast<Index>(t.row());
    const Index v = static_cast<Index>(t.col());
    if (u < 0 || v < 0 || u >= n || v >= n) {
      throw std::invalid_argument("Triplet index out of range");
    }
    row_counts[static_cast<size_t>(u)] += 1;
    if (symmetrize && u != v) {
      row_counts[static_cast<size_t>(v)] += 1;
    }
  }

  std::vector<Index> row_ptr(static_cast<size_t>(n + 1), 0);
  for (Index i = 0; i < n; ++i) {
    row_ptr[static_cast<size_t>(i + 1)] =
        row_ptr[static_cast<size_t>(i)] + row_counts[static_cast<size_t>(i)];
  }

  const Index nnz = row_ptr.back();
  std::vector<Index> col_idx(static_cast<size_t>(nnz));
  std::vector<Scalar> values(static_cast<size_t>(nnz));
  std::vector<Index> offsets = row_ptr;

  auto push_edge = [&](Index u, Index v, Scalar w) {
    const Index pos = offsets[static_cast<size_t>(u)]++;
    col_idx[static_cast<size_t>(pos)] = v;
    values[static_cast<size_t>(pos)] = w;
  };

  for (const auto& t : triplets) {
    const Index u = static_cast<Index>(t.row());
    const Index v = static_cast<Index>(t.col());
    const Scalar w = static_cast<Scalar>(t.value());
    push_edge(u, v, w);
    if (symmetrize && u != v) {
      push_edge(v, u, w);
    }
  }

  return WeightedCsrGraph(n, std::move(row_ptr), std::move(col_idx),
                          std::move(values));
}

WeightedCsrGraph WeightedCsrGraph::FromCsr(Index n,
                                           std::vector<Index> row_ptr,
                                           std::vector<Index> col_idx,
                                           std::vector<Scalar> values) {
  return WeightedCsrGraph(n, std::move(row_ptr), std::move(col_idx),
                          std::move(values));
}

const Vector& WeightedCsrGraph::degree() const {
  // Return cached degree vector.
  EnsureDegree();
  return degree_;
}

void WeightedCsrGraph::matvec(const Eigen::Ref<const Vector>& x, Vector& y) const {
  // Multiply adjacency matrix by a dense vector.
  if (x.size() != n_) {
    throw std::invalid_argument("matvec input size mismatch");
  }
  y.setZero(n_);
  for (Index row = 0; row < n_; ++row) {
    const Index start = row_ptr_[static_cast<size_t>(row)];
    const Index end = row_ptr_[static_cast<size_t>(row + 1)];
    Scalar sum = 0.0;
    for (Index idx = start; idx < end; ++idx) {
      sum += values_[static_cast<size_t>(idx)] *
             x[col_idx_[static_cast<size_t>(idx)]];
    }
    y[row] = sum;
  }
}

void WeightedCsrGraph::laplacian_matvec(const Eigen::Ref<const Vector>& x,
                                        Vector& y) const {
  // Multiply graph Laplacian by a dense vector.
  if (x.size() != n_) {
    throw std::invalid_argument("laplacian_matvec input size mismatch");
  }
  EnsureDegree();
  y.setZero(n_);
  for (Index row = 0; row < n_; ++row) {
    const Index start = row_ptr_[static_cast<size_t>(row)];
    const Index end = row_ptr_[static_cast<size_t>(row + 1)];
    Scalar sum = 0.0;
    for (Index idx = start; idx < end; ++idx) {
      sum += values_[static_cast<size_t>(idx)] *
             x[col_idx_[static_cast<size_t>(idx)]];
    }
    y[row] = degree_[row] * x[row] - sum;
  }
}

void WeightedCsrGraph::EnsureDegree() const {
  // Compute degree vector if not cached.
  if (degree_ready_) {
    return;
  }
  degree_.setZero(n_);
  for (Index row = 0; row < n_; ++row) {
    const Index start = row_ptr_[static_cast<size_t>(row)];
    const Index end = row_ptr_[static_cast<size_t>(row + 1)];
    Scalar sum = 0.0;
    for (Index idx = start; idx < end; ++idx) {
      sum += values_[static_cast<size_t>(idx)];
    }
    degree_[row] = sum;
  }
  degree_ready_ = true;
}

double WeightedCsrGraph::total_weight() const {
  // Sum all stored edge weights.
  return std::accumulate(values_.begin(), values_.end(), 0.0);
}

}  // namespace fj
