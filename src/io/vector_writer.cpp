#include "fj/io/vector_writer.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace fj {

void VectorWriter::WriteVector(const std::string& path, const Vector& values) {
  // Keep the format compatible with VectorReader and data-mode vector files.
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("Failed to open vector output: " + path);
  }
  output << std::setprecision(17);
  for (Index index = 0; index < values.size(); ++index) {
    output << values[index] << '\n';
  }
  if (!output) {
    throw std::runtime_error("Failed while writing vector output: " + path);
  }
}

}  // namespace fj
