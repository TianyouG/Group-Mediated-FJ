#pragma once

#include "fj/operators/linear_operator.hpp"
#include "fj/preconditioner/preconditioner.hpp"
#include "fj/solver/linear_solver.hpp"

namespace fj {

// Aggregated statistics for inner solves.
struct InnerSolveStats {
  // Number of Solve calls.
  Index solves = 0;
  // Total iterations across solves.
  Index iterations = 0;
  // Total time across solves (seconds).
  double seconds = 0.0;
};

class InnerSolver {
 public:
  virtual ~InnerSolver() = default;
  // Solve the inner system.
  virtual void Solve(const Vector& rhs, Vector& sol) const = 0;
  // Return the dimension of the inner system.
  virtual Index size() const = 0;
  // Reset aggregated statistics.
  virtual void ResetStats() const = 0;
  // Get aggregated statistics.
  virtual InnerSolveStats Stats() const = 0;
};

class DiagonalInnerSolver : public InnerSolver {
 public:
  // Construct a diagonal solver with a diagonal vector.
  explicit DiagonalInnerSolver(const Vector& diag);

  // Solve using element-wise division with fallbacks for zeros.
  void Solve(const Vector& rhs, Vector& sol) const override;
  // Return system size.
  Index size() const override { return static_cast<Index>(diag_.size()); }
  // Reset stats.
  void ResetStats() const override;
  // Return stats.
  InnerSolveStats Stats() const override;

 private:
  Vector diag_;
  mutable InnerSolveStats stats_;
};

class IterativeInnerSolver : public InnerSolver {
 public:
  // Construct an inner solver that delegates to a LinearSolver.
  IterativeInnerSolver(const LinearOperator& A, const LinearSolver& solver,
                       const Preconditioner* M);

  // Solve using the configured iterative solver.
  void Solve(const Vector& rhs, Vector& sol) const override;
  // Return system size.
  Index size() const override { return size_; }
  // Reset stats.
  void ResetStats() const override;
  // Return stats.
  InnerSolveStats Stats() const override;

 private:
  const LinearOperator* A_;
  const LinearSolver* solver_;
  const Preconditioner* M_;
  Index size_;
  mutable InnerSolveStats stats_;
};

}  // namespace fj
