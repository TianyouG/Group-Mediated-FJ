#pragma once

#include "fj/experiment/experiment_config.hpp"
#include "fj/experiment/experiment_result.hpp"

namespace fj {

class ExperimentRunner {
 public:
  // Initialize runner with a configuration.
  explicit ExperimentRunner(const ExperimentConfig& config);

  // Build an instance and run one experiment.
  ExperimentResult Run();

 private:
  ExperimentConfig config_;
};

}  // namespace fj
