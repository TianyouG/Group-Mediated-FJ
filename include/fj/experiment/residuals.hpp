#pragma once

#include "fj/experiment/experiment_config.hpp"
#include "fj/experiment/experiment_instance.hpp"

namespace fj {

// Compute the relative residual on the user-side Schur complement.
double SchurRelativeResidual(const ExperimentInstance& instance,
                             const ExperimentConfig& config,
                             const Vector& x_u);

}  // namespace fj
