#include "fj/preconditioner/jacobi_preconditioner.hpp"

#include <stdexcept>

namespace fj {

// Initialize Jacobi preconditioner with a diagonal vector.
JacobiPreconditioner::JacobiPreconditioner(const Vector& diag) { SetDiagonal(diag); }

void JacobiPreconditioner::SetDiagonal(const Vector& diag) {
  // Store the diagonal vector with basic validation.
  if (diag.size() == 0) {
    throw std::invalid_argument("Jacobi diagonal cannot be empty");
  }
  diag_ = diag;
}

void JacobiPreconditioner::Apply(const Vector& r, Vector& z) const {
  // Apply element-wise inverse scaling.
  if (r.size() != diag_.size()) {
    throw std::invalid_argument("JacobiPreconditioner size mismatch");
  }
  z.resize(r.size());
  for (Index i = 0; i < r.size(); ++i) {
    if (diag_[i] != 0.0) {
      z[i] = r[i] / diag_[i];
    } else {
      z[i] = r[i];
    }
  }
}

}  // namespace fj
