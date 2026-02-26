#pragma once

#include "fj/experiment/experiment_instance.hpp"

namespace fj {

// Build the diagonal of Agg = Lambda_g + D_g^c + L_g.
Vector BuildAggDiagonal(const ExperimentInstance& instance);

// Build the Jacobi diagonal for the full-system operator.
Vector BuildFullJacobiDiagonal(const ExperimentInstance& instance);

// Build the Jacobi diagonal for the Schur complement operator.
Vector BuildSchurJacobiDiagonal(const ExperimentInstance& instance);

}  // namespace fj
