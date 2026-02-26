#pragma once

#include "fj/common/types.hpp"
#include "fj/graph/csr_graph.hpp"
#include "fj/operators/linear_operator.hpp"

namespace fj {

// Compute relative residual ||b - A x|| / ||b||.
double RelativeResidual(const LinearOperator& A, const Vector& x,
                        const Vector& b);

// Compute x^T L x for a graph Laplacian.
double GraphDisagreement(const WeightedCsrGraph& graph, const Vector& x);

// Compute ||x - s||^2 for internal conflict.
double InternalConflict(const Vector& x, const Vector& s);

// Compute sum_i (x_i - mean(x))^2.
double Polarization(const Vector& x);

// Compute ||x||^2.
double Controversy(const Vector& x);

// Compute population variance of a vector.
double Variance(const Vector& x);

}  // namespace fj
