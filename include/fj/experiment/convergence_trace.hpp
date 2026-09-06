#pragma once

#include <string>

#include "fj/experiment/experiment_result.hpp"
#include "fj/solver/inner_solver.hpp"
#include "fj/solver/solver_stats.hpp"

namespace fj {

// Append one linear solve's optional history to an experiment result.
inline void AppendSolverTrace(const std::string& level, Index solve_id,
                              const SolverStats& stats,
                              ExperimentResult& result) {
  for (const ResidualTracePoint& point : stats.residual_history) {
    result.convergence_trace.push_back(
        {level, solve_id, point.iteration, point.relative_residual,
         point.seconds});
  }
}

// Append all optional inner-solve histories to an experiment result.
inline void AppendInnerTrace(const InnerSolveStats& stats,
                             ExperimentResult& result) {
  for (const InnerResidualTracePoint& point : stats.residual_history) {
    result.convergence_trace.push_back(
        {"inner", point.solve_id, point.iteration, point.relative_residual,
         point.seconds});
  }
}

}  // namespace fj

