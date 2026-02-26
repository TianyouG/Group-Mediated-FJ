#pragma once

#include <string>
#include <vector>

#include "fj/common/types.hpp"
#include "fj/graph/bipartite_csr.hpp"
#include "fj/graph/csr_graph.hpp"

namespace fj {

struct EdgeListOptions {
  // Whether indices in the file start from 1.
  bool one_indexed = false;
  // Whether to symmetrize edges when building an undirected graph.
  bool symmetrize = false;
  // Whether to drop self-loops while parsing.
  bool ignore_self_loops = true;
};

class EdgeListReader {
 public:
  // Read a weighted graph with an explicit node count.
  static WeightedCsrGraph ReadWeightedGraph(const std::string& path,
                                            Index num_nodes,
                                            const EdgeListOptions& opts);

  // Read a weighted graph and infer the node count.
  static WeightedCsrGraph ReadWeightedGraph(const std::string& path,
                                            const EdgeListOptions& opts);

  // Read a bipartite graph with explicit sizes.
  static BipartiteCsr ReadBipartite(const std::string& path,
                                    Index num_users,
                                    Index num_groups,
                                    const EdgeListOptions& opts);

  // Read a bipartite graph and infer sizes.
  static BipartiteCsr ReadBipartite(const std::string& path,
                                    const EdgeListOptions& opts);

 private:
  struct EdgeRecord {
    Index u = 0;
    Index v = 0;
    double w = 1.0;
  };

  // Parse a single edge record from a line.
  static bool ParseEdgeLine(const std::string& line, EdgeRecord& out);
};

}  // namespace fj
