#pragma once

#include <cstdint>

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace fj {

using Index = std::int64_t;
using Scalar = double;
using Vector = Eigen::VectorXd;
using SparseMatrix = Eigen::SparseMatrix<Scalar, Eigen::RowMajor, Index>;
using Triplet = Eigen::Triplet<Scalar, Index>;

}  // namespace fj
