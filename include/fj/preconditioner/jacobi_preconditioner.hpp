#pragma once

#include "fj/preconditioner/preconditioner.hpp"

namespace fj {

class JacobiPreconditioner : public Preconditioner {
 public:
  // Default constructor leaves the diagonal empty.
  JacobiPreconditioner() = default;
  // Construct with a diagonal vector.
  explicit JacobiPreconditioner(const Vector& diag);

  // Update the diagonal vector.
  void SetDiagonal(const Vector& diag);

  // Apply Jacobi preconditioning z = D^{-1} r.
  void Apply(const Vector& r, Vector& z) const override;
  // Return the size of the preconditioner.
  Index size() const override { return static_cast<Index>(diag_.size()); }

 private:
  Vector diag_;
};

}  // namespace fj
