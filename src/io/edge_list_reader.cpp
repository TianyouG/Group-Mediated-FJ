#include "fj/io/edge_list_reader.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fj {

namespace {

bool IsSkippableLine(const std::string& line) {
  // Skip empty lines and comment lines that start with '#' or '%'.
  for (char c : line) {
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
      continue;
    }
    if (c == '#' || c == '%') {
      return true;
    }
    break;
  }
  return false;
}

void ReplaceDelimiters(std::string& line) {
  // Normalize commas/tabs to spaces for simple parsing.
  std::replace(line.begin(), line.end(), ',', ' ');
  std::replace(line.begin(), line.end(), '\t', ' ');
}

}  // namespace

bool EdgeListReader::ParseEdgeLine(const std::string& line, EdgeRecord& out) {
  // Parse "u v [w]" with flexible delimiters.
  std::string work = line;
  ReplaceDelimiters(work);
  std::istringstream iss(work);
  long long u = 0;
  long long v = 0;
  if (!(iss >> u >> v)) {
    return false;
  }
  double w = 1.0;
  if (iss >> w) {
    // optional weight parsed
  }
  out.u = static_cast<Index>(u);
  out.v = static_cast<Index>(v);
  out.w = w;
  return true;
}

WeightedCsrGraph EdgeListReader::ReadWeightedGraph(const std::string& path,
                                                   Index num_nodes,
                                                   const EdgeListOptions& opts) {
  // Stream edge list to build CSR without storing all triplets.
  std::ifstream in(path);
  if (!in.is_open()) {
    throw std::runtime_error("Failed to open edge list: " + path);
  }

  std::vector<Index> row_counts;
  if (num_nodes > 0) {
    row_counts.assign(static_cast<size_t>(num_nodes), 0);
  }

  EdgeRecord edge;
  Index max_idx = -1;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || IsSkippableLine(line)) {
      continue;
    }
    if (!ParseEdgeLine(line, edge)) {
      continue;
    }

    Index u = edge.u;
    Index v = edge.v;

    if (opts.one_indexed) {
      u -= 1;
      v -= 1;
    }

    if (u < 0 || v < 0) {
      throw std::runtime_error("Edge list contains negative index");
    }

    if (opts.ignore_self_loops && u == v) {
      continue;
    }

    if (num_nodes > 0) {
      if (u >= num_nodes || v >= num_nodes) {
        throw std::runtime_error("Edge index out of range for num_nodes");
      }
    } else {
      max_idx = std::max(max_idx, std::max(u, v));
      const Index max_id = max_idx;
      if (max_id >= static_cast<Index>(row_counts.size())) {
        row_counts.resize(static_cast<size_t>(max_id + 1), 0);
      }
    }

    row_counts[static_cast<size_t>(u)] += 1;
    if (opts.symmetrize && u != v) {
      row_counts[static_cast<size_t>(v)] += 1;
    }
  }

  const Index n = num_nodes > 0 ? num_nodes : (max_idx + 1);
  if (n <= 0) {
    throw std::runtime_error("No valid edges found for graph");
  }
  if (row_counts.size() < static_cast<size_t>(n)) {
    row_counts.resize(static_cast<size_t>(n), 0);
  }

  std::vector<Index> row_ptr(static_cast<size_t>(n + 1), 0);
  for (Index i = 0; i < n; ++i) {
    row_ptr[static_cast<size_t>(i + 1)] =
        row_ptr[static_cast<size_t>(i)] + row_counts[static_cast<size_t>(i)];
  }

  const Index nnz = row_ptr.back();
  std::vector<Index> col_idx(static_cast<size_t>(nnz));
  std::vector<Scalar> values(static_cast<size_t>(nnz));
  std::vector<Index> offsets = row_ptr;

  in.clear();
  in.seekg(0, std::ios::beg);

  auto push_edge = [&](Index u, Index v, double w) {
    const Index pos = offsets[static_cast<size_t>(u)]++;
    col_idx[static_cast<size_t>(pos)] = v;
    values[static_cast<size_t>(pos)] = w;
  };

  while (std::getline(in, line)) {
    if (line.empty() || IsSkippableLine(line)) {
      continue;
    }
    if (!ParseEdgeLine(line, edge)) {
      continue;
    }

    Index u = edge.u;
    Index v = edge.v;
    double w = edge.w;

    if (opts.one_indexed) {
      u -= 1;
      v -= 1;
    }

    if (u < 0 || v < 0) {
      throw std::runtime_error("Edge list contains negative index");
    }

    if (opts.ignore_self_loops && u == v) {
      continue;
    }

    if (num_nodes > 0) {
      if (u >= num_nodes || v >= num_nodes) {
        throw std::runtime_error("Edge index out of range for num_nodes");
      }
    }

    push_edge(u, v, w);
    if (opts.symmetrize && u != v) {
      push_edge(v, u, w);
    }
  }

  return WeightedCsrGraph::FromCsr(n, std::move(row_ptr), std::move(col_idx),
                                   std::move(values));
}

WeightedCsrGraph EdgeListReader::ReadWeightedGraph(const std::string& path,
                                                   const EdgeListOptions& opts) {
  // Read edge list and infer the number of nodes.
  return ReadWeightedGraph(path, 0, opts);
}

BipartiteCsr EdgeListReader::ReadBipartite(const std::string& path,
                                           Index num_users,
                                           Index num_groups,
                                           const EdgeListOptions& opts) {
  // Stream edge list into a bipartite CSR matrix with optional size checking.
  std::ifstream in(path);
  if (!in.is_open()) {
    throw std::runtime_error("Failed to open edge list: " + path);
  }

  std::vector<Index> row_counts;
  if (num_users > 0) {
    row_counts.assign(static_cast<size_t>(num_users), 0);
  }
  EdgeRecord edge;
  Index max_u = -1;
  Index max_g = -1;

  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || IsSkippableLine(line)) {
      continue;
    }
    if (!ParseEdgeLine(line, edge)) {
      continue;
    }

    Index u = edge.u;
    Index g = edge.v;
    double w = edge.w;

    if (opts.one_indexed) {
      u -= 1;
      g -= 1;
    }

    if (u < 0 || g < 0) {
      throw std::runtime_error("Edge list contains negative index");
    }

    if (opts.ignore_self_loops && u == g) {
      // For bipartite, this only matters if indices overlap in meaning.
      continue;
    }

    if (num_users > 0 && u >= num_users) {
      throw std::runtime_error("User index out of range");
    }
    if (num_groups > 0 && g >= num_groups) {
      throw std::runtime_error("Group index out of range");
    }

    if (num_users <= 0) {
      max_u = std::max(max_u, u);
    }
    if (num_groups <= 0) {
      max_g = std::max(max_g, g);
    }

    if (num_users <= 0) {
      if (u >= static_cast<Index>(row_counts.size())) {
        row_counts.resize(static_cast<size_t>(u + 1), 0);
      }
    }
    row_counts[static_cast<size_t>(u)] += 1;
  }

  Index n_users = num_users > 0 ? num_users : (max_u + 1);
  Index n_groups = num_groups > 0 ? num_groups : (max_g + 1);
  if (n_users <= 0 || n_groups <= 0) {
    throw std::runtime_error("No valid edges found for bipartite graph");
  }

  if (row_counts.size() < static_cast<size_t>(n_users)) {
    row_counts.resize(static_cast<size_t>(n_users), 0);
  }

  std::vector<Index> row_ptr(static_cast<size_t>(n_users + 1), 0);
  for (Index u = 0; u < n_users; ++u) {
    row_ptr[static_cast<size_t>(u + 1)] =
        row_ptr[static_cast<size_t>(u)] + row_counts[static_cast<size_t>(u)];
  }

  const Index nnz = row_ptr.back();
  std::vector<Index> col_idx(static_cast<size_t>(nnz));
  std::vector<Scalar> values(static_cast<size_t>(nnz));
  std::vector<Index> offsets = row_ptr;

  in.clear();
  in.seekg(0, std::ios::beg);

  auto push_edge = [&](Index u, Index g, double w) {
    const Index pos = offsets[static_cast<size_t>(u)]++;
    col_idx[static_cast<size_t>(pos)] = g;
    values[static_cast<size_t>(pos)] = w;
  };

  while (std::getline(in, line)) {
    if (line.empty() || IsSkippableLine(line)) {
      continue;
    }
    if (!ParseEdgeLine(line, edge)) {
      continue;
    }

    Index u = edge.u;
    Index g = edge.v;
    double w = edge.w;

    if (opts.one_indexed) {
      u -= 1;
      g -= 1;
    }

    if (u < 0 || g < 0) {
      throw std::runtime_error("Edge list contains negative index");
    }

    if (opts.ignore_self_loops && u == g) {
      continue;
    }

    if (num_users > 0 && u >= num_users) {
      throw std::runtime_error("User index out of range");
    }
    if (num_groups > 0 && g >= num_groups) {
      throw std::runtime_error("Group index out of range");
    }

    push_edge(u, g, w);
  }

  return BipartiteCsr::FromCsr(n_users, n_groups, std::move(row_ptr),
                               std::move(col_idx), std::move(values));
}

BipartiteCsr EdgeListReader::ReadBipartite(const std::string& path,
                                           const EdgeListOptions& opts) {
  // Read bipartite edge list and infer the user/group sizes.
  return ReadBipartite(path, 0, 0, opts);
}

}  // namespace fj
