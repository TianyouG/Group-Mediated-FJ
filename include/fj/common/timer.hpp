#pragma once

#include <chrono>

namespace fj {

class Timer {
 public:
  // Start the timer at construction.
  Timer() { Reset(); }

  // Reset the start time to now.
  void Reset() { start_ = Clock::now(); }

  // Return elapsed time in seconds.
  double ElapsedSeconds() const {
    return std::chrono::duration<double>(Clock::now() - start_).count();
  }

 private:
  using Clock = std::chrono::steady_clock;
  Clock::time_point start_;
};

}  // namespace fj
