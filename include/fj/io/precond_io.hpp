#pragma once

#include <cstdint>
#include <string>

#include "fj/common/types.hpp"
#include "fj/graph/bipartite_csr.hpp"
#include "fj/graph/csr_graph.hpp"

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
};

struct PrecondMetadata {
  std::int64_t n_users;
  std::int64_t n_groups;
  std::int64_t user_graph_nnz;
  std::int64_t group_graph_nnz;
  std::int64_t bipartite_nnz;
  double user_graph_weight_sum;
  double group_graph_weight_sum;
  double bipartite_weight_sum;
  double lambda_user;
  double lambda_group;
  double user_graph_scale;
  double group_graph_scale;
};

class PrecondIO {
 public:
  static constexpr std::uint32_t kVersion = 2;

  // Build cache metadata from the exact scaled graphs used by a solver.
  static PrecondMetadata MakeMetadata(
      const WeightedCsrGraph& user_graph,
      const WeightedCsrGraph& group_graph,
      const BipartiteCsr& bipartite, double lambda_user,
      double lambda_group, double user_graph_scale,
      double group_graph_scale);

  // Write a Jacobi diagonal vector with metadata.
  static void WriteJacobiDiag(const std::string& path, PrecondKind kind,
                              const Vector& diag,
                              const PrecondMetadata& metadata);

  // Read a Jacobi diagonal vector, validating metadata.
  static Vector ReadJacobiDiag(const std::string& path, PrecondKind kind,
                               Index expected_size,
                               const PrecondMetadata& expected_metadata);
};

}  // namespace fj
