#pragma once

#include "fj/common/types.hpp"

namespace fj {

class Preconditioner {
 public:
  virtual ~Preconditioner() = default;
  // Apply the preconditioner to r and write into z.
  virtual void Apply(const Vector& r, Vector& z) const = 0;
  // Return the size of the preconditioner.
  virtual Index size() const = 0;
};

}  // namespace fj
