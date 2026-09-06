#pragma once

namespace fj {

// Return the process peak resident-set size in MiB, or NaN if unavailable.
double PeakResidentMemoryMb();

}  // namespace fj

