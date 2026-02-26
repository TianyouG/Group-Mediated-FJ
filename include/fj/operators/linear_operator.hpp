#pragma once

#include "fj/common/types.hpp"

namespace fj {

class LinearOperator {
 public:
  virtual ~LinearOperator() = default;

  // Return number of rows.
  virtual Index rows() const = 0;
  // Return number of cols.
  virtual Index cols() const = 0;
  // Apply the operator to x and write into y.
  virtual void Apply(const Vector& x, Vector& y) const = 0;

  // Convenience wrapper that allocates and returns the result.
  Vector Apply(const Vector& x) const {
    Vector y(rows());
    Apply(x, y);
    return y;
  }
};

}  // namespace fj
