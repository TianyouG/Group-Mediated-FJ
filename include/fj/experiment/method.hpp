#pragma once

#include <string>

namespace fj {

enum class Method {
  Schur = 0,
  FullSystem = 1,
  Clique = 2,
  Direct = 3,
  FjDynamics = 4,
};

// Parse a method name from string.
Method ParseMethod(const std::string& name);
// Convert a method enum to string.
std::string ToString(Method method);

}  // namespace fj
