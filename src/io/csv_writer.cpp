#include "fj/io/csv_writer.hpp"

#include <sstream>
#include <stdexcept>

namespace fj {

CsvWriter::CsvWriter(const std::string& path) : out_(path) {
  // Open output file immediately to fail fast on errors.
  if (!out_.is_open()) {
    throw std::runtime_error("Failed to open CSV file: " + path);
  }
}

void CsvWriter::WriteHeader(const std::vector<std::string>& cols) {
  // Header is written like a normal row.
  WriteRowInternal(cols);
}

void CsvWriter::WriteRow(const std::vector<std::string>& cols) {
  // Write a row of pre-formatted strings.
  WriteRowInternal(cols);
}

void CsvWriter::WriteRow(const std::vector<double>& cols) {
  // Format numbers with full precision before writing.
  std::vector<std::string> str(cols.size());
  for (size_t i = 0; i < cols.size(); ++i) {
    std::ostringstream os;
    os.precision(17);
    os << cols[i];
    str[i] = os.str();
  }
  WriteRowInternal(str);
}

void CsvWriter::WriteRowInternal(const std::vector<std::string>& cols) {
  // Emit a single CSV row with comma separators.
  for (size_t i = 0; i < cols.size(); ++i) {
    out_ << cols[i];
    if (i + 1 < cols.size()) {
      out_ << ',';
    }
  }
  out_ << '\n';
}

}  // namespace fj
