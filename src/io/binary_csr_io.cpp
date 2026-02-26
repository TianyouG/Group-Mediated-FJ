#include "fj/io/binary_csr_io.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace fj {
namespace {

constexpr char kCsrMagic[8] = {'F', 'J', 'C', 'S', 'R', '0', '1', '\0'};

void WriteBytes(std::ofstream& out, const void* data, std::size_t len) {
  out.write(reinterpret_cast<const char*>(data),
            static_cast<std::streamsize>(len));
  if (!out) {
    throw std::runtime_error("Failed to write binary CSR file");
  }
}

void ReadBytes(std::ifstream& in, void* data, std::size_t len) {
  in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(len));
  if (!in) {
    throw std::runtime_error("Failed to read binary CSR file");
  }
}

template <typename T>
void WriteScalar(std::ofstream& out, const T& value) {
  WriteBytes(out, &value, sizeof(T));
}

template <typename T>
T ReadScalar(std::ifstream& in) {
  T value{};
  ReadBytes(in, &value, sizeof(T));
  return value;
}

CsrBinaryHeader MakeHeader(std::uint32_t flags, std::int64_t n_rows,
                           std::int64_t n_cols, std::int64_t nnz) {
  CsrBinaryHeader header{};
  std::memcpy(header.magic, kCsrMagic, sizeof(header.magic));
  header.version = BinaryCsrIO::kVersion;
  header.flags = flags;
  header.n_rows = n_rows;
  header.n_cols = n_cols;
  header.nnz = nnz;
  return header;
}

CsrBinaryHeader ReadHeader(std::ifstream& in) {
  CsrBinaryHeader header{};
  ReadBytes(in, header.magic, sizeof(header.magic));
  header.version = ReadScalar<std::uint32_t>(in);
  header.flags = ReadScalar<std::uint32_t>(in);
  header.n_rows = ReadScalar<std::int64_t>(in);
  header.n_cols = ReadScalar<std::int64_t>(in);
  header.nnz = ReadScalar<std::int64_t>(in);
  if (std::memcmp(header.magic, kCsrMagic, sizeof(header.magic)) != 0) {
    throw std::runtime_error("Invalid CSR binary header magic");
  }
  if (header.version != BinaryCsrIO::kVersion) {
    throw std::runtime_error("Unsupported CSR binary version");
  }
  if (header.n_rows < 0 || header.n_cols < 0 || header.nnz < 0) {
    throw std::runtime_error("Invalid CSR binary header sizes");
  }
  return header;
}

void WriteHeader(std::ofstream& out, const CsrBinaryHeader& header) {
  WriteBytes(out, header.magic, sizeof(header.magic));
  WriteScalar(out, header.version);
  WriteScalar(out, header.flags);
  WriteScalar(out, header.n_rows);
  WriteScalar(out, header.n_cols);
  WriteScalar(out, header.nnz);
}

template <typename T>
std::vector<T> ReadVector(std::ifstream& in, std::size_t count) {
  std::vector<T> out(count);
  if (count == 0) {
    return out;
  }
  ReadBytes(in, out.data(), sizeof(T) * count);
  return out;
}

}  // namespace

void BinaryCsrIO::WriteWeightedGraph(const std::string& path,
                                     const WeightedCsrGraph& graph) {
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    throw std::runtime_error("Failed to open binary CSR file: " + path);
  }

  const std::int64_t n = static_cast<std::int64_t>(graph.num_nodes());
  const std::int64_t nnz = static_cast<std::int64_t>(graph.nnz());
  CsrBinaryHeader header = MakeHeader(0, n, n, nnz);
  WriteHeader(out, header);

  const auto& row_ptr = graph.row_ptr();
  const auto& col_idx = graph.col_idx();
  const auto& values = graph.values();
  if (row_ptr.size() != static_cast<std::size_t>(n + 1) ||
      col_idx.size() != static_cast<std::size_t>(nnz) ||
      values.size() != static_cast<std::size_t>(nnz)) {
    throw std::runtime_error("CSR buffers size mismatch while writing");
  }

  WriteBytes(out, row_ptr.data(), sizeof(Index) * row_ptr.size());
  WriteBytes(out, col_idx.data(), sizeof(Index) * col_idx.size());
  WriteBytes(out, values.data(), sizeof(Scalar) * values.size());
}

void BinaryCsrIO::WriteBipartite(const std::string& path,
                                 const BipartiteCsr& bipartite) {
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    throw std::runtime_error("Failed to open binary CSR file: " + path);
  }

  const std::int64_t n_users = static_cast<std::int64_t>(bipartite.num_users());
  const std::int64_t n_groups = static_cast<std::int64_t>(bipartite.num_groups());
  const std::int64_t nnz = static_cast<std::int64_t>(bipartite.nnz());
  CsrBinaryHeader header =
      MakeHeader(kFlagBipartite, n_users, n_groups, nnz);
  WriteHeader(out, header);

  const auto& row_ptr = bipartite.row_ptr();
  const auto& col_idx = bipartite.col_idx();
  const auto& values = bipartite.values();
  if (row_ptr.size() != static_cast<std::size_t>(n_users + 1) ||
      col_idx.size() != static_cast<std::size_t>(nnz) ||
      values.size() != static_cast<std::size_t>(nnz)) {
    throw std::runtime_error("CSR buffers size mismatch while writing");
  }

  WriteBytes(out, row_ptr.data(), sizeof(Index) * row_ptr.size());
  WriteBytes(out, col_idx.data(), sizeof(Index) * col_idx.size());
  WriteBytes(out, values.data(), sizeof(Scalar) * values.size());
}

WeightedCsrGraph BinaryCsrIO::ReadWeightedGraph(const std::string& path,
                                                Index expected_nodes) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Failed to open binary CSR file: " + path);
  }

  CsrBinaryHeader header = ReadHeader(in);
  if (header.flags & kFlagBipartite) {
    throw std::runtime_error("Expected square CSR but found bipartite flag");
  }
  if (header.n_rows != header.n_cols) {
    throw std::runtime_error("CSR binary graph must be square");
  }
  if (expected_nodes > 0 && header.n_rows != expected_nodes) {
    throw std::runtime_error("CSR binary graph size mismatch");
  }

  const std::size_t rows = static_cast<std::size_t>(header.n_rows);
  const std::size_t nnz = static_cast<std::size_t>(header.nnz);

  std::vector<Index> row_ptr = ReadVector<Index>(in, rows + 1);
  std::vector<Index> col_idx = ReadVector<Index>(in, nnz);
  std::vector<Scalar> values = ReadVector<Scalar>(in, nnz);
  return WeightedCsrGraph::FromCsr(static_cast<Index>(header.n_rows),
                                   std::move(row_ptr), std::move(col_idx),
                                   std::move(values));
}

BipartiteCsr BinaryCsrIO::ReadBipartite(const std::string& path,
                                        Index expected_users,
                                        Index expected_groups) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Failed to open binary CSR file: " + path);
  }

  CsrBinaryHeader header = ReadHeader(in);
  if (!(header.flags & kFlagBipartite)) {
    throw std::runtime_error("Expected bipartite CSR but flag is missing");
  }
  if (expected_users > 0 && header.n_rows != expected_users) {
    throw std::runtime_error("Bipartite CSR user size mismatch");
  }
  if (expected_groups > 0 && header.n_cols != expected_groups) {
    throw std::runtime_error("Bipartite CSR group size mismatch");
  }

  const std::size_t rows = static_cast<std::size_t>(header.n_rows);
  const std::size_t nnz = static_cast<std::size_t>(header.nnz);

  std::vector<Index> row_ptr = ReadVector<Index>(in, rows + 1);
  std::vector<Index> col_idx = ReadVector<Index>(in, nnz);
  std::vector<Scalar> values = ReadVector<Scalar>(in, nnz);
  return BipartiteCsr::FromCsr(static_cast<Index>(header.n_rows),
                               static_cast<Index>(header.n_cols),
                               std::move(row_ptr), std::move(col_idx),
                               std::move(values));
}

}  // namespace fj
