#pragma once

#include <string>

#include "fj/common/types.hpp"

namespace fj {

class VectorWriter {
 public:
  // Write one vector value per line with round-trip precision.
  static void WriteVector(const std::string& path, const Vector& values);
};

}  // namespace fj

