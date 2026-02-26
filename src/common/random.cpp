#include "fj/common/random.hpp"

namespace fj {

// Initialize RNG state.
Random::Random(uint64_t seed) : rng_(seed) {}

// Reseed RNG.
void Random::Seed(uint64_t seed) { rng_.seed(seed); }

double Random::Uniform01() {
  // Sample uniform random in [0, 1].
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(rng_);
}

Index Random::UniformInt(Index lo, Index hi) {
  // Sample uniform integer in [lo, hi].
  std::uniform_int_distribution<Index> dist(lo, hi);
  return dist(rng_);
}

double Random::Normal(double mean, double stddev) {
  // Sample normal random value.
  std::normal_distribution<double> dist(mean, stddev);
  return dist(rng_);
}

}  // namespace fj
