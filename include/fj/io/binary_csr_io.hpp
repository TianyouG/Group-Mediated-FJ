#pragma once

#include <cstdint>
#include <string>

#include "fj/common/types.hpp"
#include "fj/graph/bipartite_csr.hpp"
#include "fj/graph/csr_graph.hpp"

namespace fj {

struct CsrBinaryHeader {
  char magic[8];
  std::uint32_t version;
  std::uint32_t flags;
  std::int64_t n_rows;
  std::int64_t n_cols;
  std::int64_t nnz;
};

class BinaryCsrIO {
 public:
  static constexpr std::uint32_t kVersion = 1;
  static constexpr std::uint32_t kFlagBipartite = 1u << 0;

  // Write a weighted square graph to a binary CSR file.
  static void WriteWeightedGraph(const std::string& path,
                                 const WeightedCsrGraph& graph);
  // Write a bipartite CSR to a binary file.
  static void WriteBipartite(const std::string& path,
                             const BipartiteCsr& bipartite);

  // Read a weighted square graph from a binary CSR file.
  static WeightedCsrGraph ReadWeightedGraph(const std::string& path,
                                            Index expected_nodes = 0);
  // Read a bipartite CSR from a binary CSR file.
  static BipartiteCsr ReadBipartite(const std::string& path,
                                    Index expected_users = 0,
                                    Index expected_groups = 0);
};

}  // namespace fj
