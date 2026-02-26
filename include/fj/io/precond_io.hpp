#pragma once

#include <cstdint>
#include <string>

#include "fj/common/types.hpp"

namespace fj {

enum class PrecondKind : std::uint32_t {
  kFullJacobi = 1,
  kSchurJacobi = 2,
  kAggJacobi = 3,
};

struct PrecondHeader {
  char magic[8];
  std::uint32_t version;
  std::uint32_t kind;
  std::int64_t size;
  std::int64_t n_users;
  std::int64_t n_groups;
  double lambda_user;
  double lambda_group;
};

class PrecondIO {
 public:
  static constexpr std::uint32_t kVersion = 1;

  // Write a Jacobi diagonal vector with metadata.
  static void WriteJacobiDiag(const std::string& path, PrecondKind kind,
                              const Vector& diag, Index n_users,
                              Index n_groups, double lambda_user,
                              double lambda_group);

  // Read a Jacobi diagonal vector, validating metadata.
  static Vector ReadJacobiDiag(const std::string& path, PrecondKind kind,
                               Index expected_size, Index n_users,
                               Index n_groups, double lambda_user,
                               double lambda_group);
};

}  // namespace fj
