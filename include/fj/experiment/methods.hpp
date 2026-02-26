#pragma once

#include "fj/experiment/experiment_config.hpp"
#include "fj/experiment/experiment_instance.hpp"
#include "fj/experiment/experiment_result.hpp"

namespace fj {

// Run the Schur-complement method.
ExperimentResult RunSchurMethod(const ExperimentInstance& instance,
                                const ExperimentConfig& config);

// Run the full-system CG/PCG baseline.
ExperimentResult RunFullSystemMethod(const ExperimentInstance& instance,
                                     const ExperimentConfig& config);

// Run the clique-expansion baseline.
ExperimentResult RunCliqueMethod(const ExperimentInstance& instance,
                                 const ExperimentConfig& config);

// Run the direct dense baseline (small problems only).
ExperimentResult RunDirectMethod(const ExperimentInstance& instance,
                                 const ExperimentConfig& config);

// Run the fixed-point FJ dynamics baseline.
ExperimentResult RunFjDynamicsMethod(const ExperimentInstance& instance,
                                     const ExperimentConfig& config);

}  // namespace fj
