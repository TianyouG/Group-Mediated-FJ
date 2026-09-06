#include "fj/io/precond_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace fj {
namespace {

constexpr char kPrecondMagic[8] = {'F', 'J', 'P', 'C', 'D', '0', '1', '\0'};

void WriteBytes(std::ofstream& out, const void* data, std::size_t len) {
  out.write(reinterpret_cast<const char*>(data),
            static_cast<std::streamsize>(len));
  if (!out) {
    throw std::runtime_error("Failed to write preconditioner file");
  }
}

void ReadBytes(std::ifstream& in, void* data, std::size_t len) {
  in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(len));
  if (!in) {
    throw std::runtime_error("Failed to read preconditioner file");
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

bool NearlyEqual(double a, double b) {
  const double diff = std::abs(a - b);
  const double scale = std::max(1.0, std::max(std::abs(a), std::abs(b)));
  return diff <= 1e-12 * scale;
}

PrecondHeader MakeHeader(PrecondKind kind, std::int64_t size) {
  PrecondHeader header{};
  std::memcpy(header.magic, kPrecondMagic, sizeof(header.magic));
  header.version = PrecondIO::kVersion;
  header.kind = static_cast<std::uint32_t>(kind);
  header.size = size;
  return header;
}

PrecondHeader ReadHeader(std::ifstream& in) {
  PrecondHeader header{};
  ReadBytes(in, header.magic, sizeof(header.magic));
  header.version = ReadScalar<std::uint32_t>(in);
  header.kind = ReadScalar<std::uint32_t>(in);
  header.size = ReadScalar<std::int64_t>(in);
  if (std::memcmp(header.magic, kPrecondMagic, sizeof(header.magic)) != 0) {
    throw std::runtime_error("Invalid preconditioner file magic");
  }
  if (header.version != PrecondIO::kVersion) {
    throw std::runtime_error("Unsupported preconditioner file version");
  }
  if (header.size < 0) {
    throw std::runtime_error("Invalid preconditioner metadata");
  }
  return header;
}

void WriteHeader(std::ofstream& out, const PrecondHeader& header) {
  WriteBytes(out, header.magic, sizeof(header.magic));
  WriteScalar(out, header.version);
  WriteScalar(out, header.kind);
  WriteScalar(out, header.size);
}

void WriteMetadata(std::ofstream& out, const PrecondMetadata& metadata) {
  WriteScalar(out, metadata.n_users);
  WriteScalar(out, metadata.n_groups);
  WriteScalar(out, metadata.user_graph_nnz);
  WriteScalar(out, metadata.group_graph_nnz);
  WriteScalar(out, metadata.bipartite_nnz);
  WriteScalar(out, metadata.user_graph_weight_sum);
  WriteScalar(out, metadata.group_graph_weight_sum);
  WriteScalar(out, metadata.bipartite_weight_sum);
  WriteScalar(out, metadata.lambda_user);
  WriteScalar(out, metadata.lambda_group);
  WriteScalar(out, metadata.user_graph_scale);
  WriteScalar(out, metadata.group_graph_scale);
}

PrecondMetadata ReadMetadata(std::ifstream& in) {
  PrecondMetadata metadata{};
  metadata.n_users = ReadScalar<std::int64_t>(in);
  metadata.n_groups = ReadScalar<std::int64_t>(in);
  metadata.user_graph_nnz = ReadScalar<std::int64_t>(in);
  metadata.group_graph_nnz = ReadScalar<std::int64_t>(in);
  metadata.bipartite_nnz = ReadScalar<std::int64_t>(in);
  metadata.user_graph_weight_sum = ReadScalar<double>(in);
  metadata.group_graph_weight_sum = ReadScalar<double>(in);
  metadata.bipartite_weight_sum = ReadScalar<double>(in);
  metadata.lambda_user = ReadScalar<double>(in);
  metadata.lambda_group = ReadScalar<double>(in);
  metadata.user_graph_scale = ReadScalar<double>(in);
  metadata.group_graph_scale = ReadScalar<double>(in);
  if (metadata.n_users < 0 || metadata.n_groups < 0 ||
      metadata.user_graph_nnz < 0 || metadata.group_graph_nnz < 0 ||
      metadata.bipartite_nnz < 0) {
    throw std::runtime_error("Invalid preconditioner graph metadata");
  }
  return metadata;
}

void ValidateMetadata(const PrecondMetadata& actual,
                      const PrecondMetadata& expected) {
  if (actual.n_users != expected.n_users ||
      actual.n_groups != expected.n_groups) {
    throw std::runtime_error("Preconditioner graph size mismatch");
  }
  if (!NearlyEqual(actual.lambda_user, expected.lambda_user) ||
      !NearlyEqual(actual.lambda_group, expected.lambda_group)) {
    throw std::runtime_error("Preconditioner lambda values mismatch");
  }
  if (!NearlyEqual(actual.user_graph_scale, expected.user_graph_scale) ||
      !NearlyEqual(actual.group_graph_scale, expected.group_graph_scale)) {
    throw std::runtime_error("Preconditioner graph scale mismatch");
  }
  if (actual.user_graph_nnz != expected.user_graph_nnz ||
      actual.group_graph_nnz != expected.group_graph_nnz ||
      actual.bipartite_nnz != expected.bipartite_nnz) {
    throw std::runtime_error("Preconditioner graph nonzero count mismatch");
  }
  if (!NearlyEqual(actual.user_graph_weight_sum,
                   expected.user_graph_weight_sum) ||
      !NearlyEqual(actual.group_graph_weight_sum,
                   expected.group_graph_weight_sum) ||
      !NearlyEqual(actual.bipartite_weight_sum,
                   expected.bipartite_weight_sum)) {
    throw std::runtime_error("Preconditioner graph weight signature mismatch");
  }
}

}  // namespace

PrecondMetadata PrecondIO::MakeMetadata(
    const WeightedCsrGraph& user_graph,
    const WeightedCsrGraph& group_graph,
    const BipartiteCsr& bipartite, double lambda_user,
    double lambda_group, double user_graph_scale,
    double group_graph_scale) {
  if (user_graph.num_nodes() != bipartite.num_users() ||
      group_graph.num_nodes() != bipartite.num_groups()) {
    throw std::invalid_argument("Preconditioner graph dimensions do not match");
  }
  PrecondMetadata metadata{};
  metadata.n_users = bipartite.num_users();
  metadata.n_groups = bipartite.num_groups();
  metadata.user_graph_nnz = user_graph.nnz();
  metadata.group_graph_nnz = group_graph.nnz();
  metadata.bipartite_nnz = bipartite.nnz();
  metadata.user_graph_weight_sum = user_graph.degree().sum();
  metadata.group_graph_weight_sum = group_graph.degree().sum();
  metadata.bipartite_weight_sum = bipartite.user_degree().sum();
  metadata.lambda_user = lambda_user;
  metadata.lambda_group = lambda_group;
  metadata.user_graph_scale = user_graph_scale;
  metadata.group_graph_scale = group_graph_scale;
  return metadata;
}

void PrecondIO::WriteJacobiDiag(const std::string& path, PrecondKind kind,
                                const Vector& diag,
                                const PrecondMetadata& metadata) {
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    throw std::runtime_error("Failed to open preconditioner file: " + path);
  }
  const std::int64_t size = static_cast<std::int64_t>(diag.size());
  PrecondHeader header = MakeHeader(kind, size);
  WriteHeader(out, header);
  WriteMetadata(out, metadata);
  if (size > 0) {
    WriteBytes(out, diag.data(), sizeof(double) * static_cast<std::size_t>(size));
  }
}

Vector PrecondIO::ReadJacobiDiag(const std::string& path, PrecondKind kind,
                                 Index expected_size,
                                 const PrecondMetadata& expected_metadata) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Failed to open preconditioner file: " + path);
  }
  PrecondHeader header = ReadHeader(in);
  if (header.kind != static_cast<std::uint32_t>(kind)) {
    throw std::runtime_error("Preconditioner kind mismatch");
  }
  if (expected_size > 0 && header.size != expected_size) {
    throw std::runtime_error("Preconditioner size mismatch");
  }
  const PrecondMetadata actual_metadata = ReadMetadata(in);
  ValidateMetadata(actual_metadata, expected_metadata);

  const std::size_t size = static_cast<std::size_t>(header.size);
  Vector diag(static_cast<Index>(size));
  if (size > 0) {
    ReadBytes(in, diag.data(), sizeof(double) * size);
  }
  return diag;
}

}  // namespace fj
