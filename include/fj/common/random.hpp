#pragma once

#include <cstdint>
#include <random>

#include "fj/common/types.hpp"

namespace fj {

class Random {
 public:
  // Construct RNG with a seed.
  explicit Random(uint64_t seed = 1);

  // Reset RNG seed.
  void Seed(uint64_t seed);

  // Uniform random in [0,1].
  double Uniform01();
  // Uniform integer in [lo, hi].
  Index UniformInt(Index lo, Index hi);
  // Normal random with mean and stddev.
  double Normal(double mean, double stddev);

 private:
  std::mt19937_64 rng_;
};

}  // namespace fj
