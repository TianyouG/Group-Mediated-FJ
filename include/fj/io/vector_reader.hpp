#pragma once

#include <string>

#include "fj/common/types.hpp"

namespace fj {

class VectorReader {
 public:
  // Read a dense vector from a file (one value per non-comment line).
  static Vector ReadVector(const std::string& path, Index expected_size);
};

}  // namespace fj
