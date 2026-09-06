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
  if (value == "bli") {
    return Method::Bli;
  }
  if (value == "bli_sor" || value == "blisor" || value == "bli-sor") {
    return Method::BliSor;
  }
  if (value == "pf_qe" || value == "pfqe" || value == "pf-qe") {
    return Method::PfQe;
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
    case Method::Bli:
      return "bli";
    case Method::BliSor:
      return "bli_sor";
    case Method::PfQe:
      return "pf_qe";
    default:
      return "unknown";
  }
}

}  // namespace fj
