#include "fj/io/vector_reader.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

Vector VectorReader::ReadVector(const std::string& path, Index expected_size) {
  // Read a numeric vector and validate its length.
  std::ifstream in(path);
  if (!in.is_open()) {
    throw std::runtime_error("Failed to open vector file: " + path);
  }

  std::vector<double> values;
  if (expected_size > 0) {
    values.reserve(static_cast<size_t>(expected_size));
  }

  std::string line;
  Index line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    if (line.empty() || IsSkippableLine(line)) {
      continue;
    }
    ReplaceDelimiters(line);
    std::istringstream iss(line);
    double v = 0.0;
    if (!(iss >> v)) {
      throw std::runtime_error("Invalid numeric value in " + path + " at line " +
                               std::to_string(line_no));
    }
    values.push_back(v);
  }

  if (values.empty()) {
    throw std::runtime_error("No values found in vector file: " + path);
  }

  if (expected_size <= 0) {
    expected_size = static_cast<Index>(values.size());
  }
  if (static_cast<Index>(values.size()) != expected_size) {
    throw std::runtime_error("Vector length mismatch for " + path +
                             ": expected " + std::to_string(expected_size) +
                             ", got " + std::to_string(values.size()));
  }

  Vector vec(expected_size);
  for (Index i = 0; i < expected_size; ++i) {
    vec[i] = values[static_cast<size_t>(i)];
  }
  return vec;
}

}  // namespace fj
