#include "fj/experiment/method.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace fj {

Method ParseMethod(const std::string& name) {
  // Parse method name in a case-insensitive way.
  std::string value = name;
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });

  if (value == "schur") {
    return Method::Schur;
  }
  if (value == "full_system" || value == "full" || value == "fullsystem") {
    return Method::FullSystem;
  }
  if (value == "clique" || value == "clique_expansion") {
    return Method::Clique;
  }
  if (value == "direct") {
    return Method::Direct;
  }
  if (value == "fj_dynamics" || value == "fjdynamics" || value == "dynamics" ||
      value == "fj") {
    return Method::FjDynamics;
  }
  throw std::invalid_argument("Unknown method: " + name);
}

std::string ToString(Method method) {
  // Convert method enum to a stable string.
  switch (method) {
    case Method::Schur:
      return "schur";
    case Method::FullSystem:
      return "full_system";
    case Method::Clique:
      return "clique";
    case Method::Direct:
      return "direct";
    case Method::FjDynamics:
      return "fj_dynamics";
    default:
      return "unknown";
  }
}

}  // namespace fj
