#pragma once

#include <fstream>
#include <string>
#include <vector>

namespace fj {

class CsvWriter {
 public:
  // Open a CSV file for writing (overwrites existing file).
  explicit CsvWriter(const std::string& path);

  // Check whether the output file is open.
  bool IsOpen() const { return out_.is_open(); }

  // Write the header row.
  void WriteHeader(const std::vector<std::string>& cols);
  // Write a string row.
  void WriteRow(const std::vector<std::string>& cols);
  // Write a numeric row with full precision.
  void WriteRow(const std::vector<double>& cols);

 private:
  std::ofstream out_;
  // Low-level row writer without additional validation.
  void WriteRowInternal(const std::vector<std::string>& cols);
};

}  // namespace fj
