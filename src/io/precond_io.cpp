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

PrecondHeader MakeHeader(PrecondKind kind, std::int64_t size, std::int64_t n_users,
                         std::int64_t n_groups, double lambda_user,
                         double lambda_group) {
  PrecondHeader header{};
  std::memcpy(header.magic, kPrecondMagic, sizeof(header.magic));
  header.version = PrecondIO::kVersion;
  header.kind = static_cast<std::uint32_t>(kind);
  header.size = size;
  header.n_users = n_users;
  header.n_groups = n_groups;
  header.lambda_user = lambda_user;
  header.lambda_group = lambda_group;
  return header;
}

PrecondHeader ReadHeader(std::ifstream& in) {
  PrecondHeader header{};
  ReadBytes(in, header.magic, sizeof(header.magic));
  header.version = ReadScalar<std::uint32_t>(in);
  header.kind = ReadScalar<std::uint32_t>(in);
  header.size = ReadScalar<std::int64_t>(in);
  header.n_users = ReadScalar<std::int64_t>(in);
  header.n_groups = ReadScalar<std::int64_t>(in);
  header.lambda_user = ReadScalar<double>(in);
  header.lambda_group = ReadScalar<double>(in);
  if (std::memcmp(header.magic, kPrecondMagic, sizeof(header.magic)) != 0) {
    throw std::runtime_error("Invalid preconditioner file magic");
  }
  if (header.version != PrecondIO::kVersion) {
    throw std::runtime_error("Unsupported preconditioner file version");
  }
  if (header.size < 0 || header.n_users < 0 || header.n_groups < 0) {
    throw std::runtime_error("Invalid preconditioner metadata");
  }
  return header;
}

void WriteHeader(std::ofstream& out, const PrecondHeader& header) {
  WriteBytes(out, header.magic, sizeof(header.magic));
  WriteScalar(out, header.version);
  WriteScalar(out, header.kind);
  WriteScalar(out, header.size);
  WriteScalar(out, header.n_users);
  WriteScalar(out, header.n_groups);
  WriteScalar(out, header.lambda_user);
  WriteScalar(out, header.lambda_group);
}

}  // namespace

void PrecondIO::WriteJacobiDiag(const std::string& path, PrecondKind kind,
                                const Vector& diag, Index n_users,
                                Index n_groups, double lambda_user,
                                double lambda_group) {
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    throw std::runtime_error("Failed to open preconditioner file: " + path);
  }
  const std::int64_t size = static_cast<std::int64_t>(diag.size());
  PrecondHeader header = MakeHeader(kind, size, n_users, n_groups, lambda_user,
                                    lambda_group);
  WriteHeader(out, header);
  if (size > 0) {
    WriteBytes(out, diag.data(), sizeof(double) * static_cast<std::size_t>(size));
  }
}

Vector PrecondIO::ReadJacobiDiag(const std::string& path, PrecondKind kind,
                                 Index expected_size, Index n_users,
                                 Index n_groups, double lambda_user,
                                 double lambda_group) {
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
  if (header.n_users != n_users || header.n_groups != n_groups) {
    throw std::runtime_error("Preconditioner graph size mismatch");
  }
  if (!NearlyEqual(header.lambda_user, lambda_user) ||
      !NearlyEqual(header.lambda_group, lambda_group)) {
    throw std::runtime_error("Preconditioner lambda values mismatch");
  }

  const std::size_t size = static_cast<std::size_t>(header.size);
  Vector diag(static_cast<Index>(size));
  if (size > 0) {
    ReadBytes(in, diag.data(), sizeof(double) * size);
  }
  return diag;
}

}  // namespace fj
