#include "fj/common/memory.hpp"

#include <limits>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

namespace fj {

double PeakResidentMemoryMb() {
  // getrusage reports KiB on Linux and bytes on macOS.
#if defined(__linux__) || defined(__APPLE__)
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
#if defined(__APPLE__)
  return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
#else
  return static_cast<double>(usage.ru_maxrss) / 1024.0;
#endif
#else
  return std::numeric_limits<double>::quiet_NaN();
#endif
}

}  // namespace fj

