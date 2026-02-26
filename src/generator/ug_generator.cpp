#include "fj/generator/ug_generator.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fj {
namespace {

void ValidateConfig(const UgConfig& cfg) {
  // Validate UG generator configuration.
  if (cfg.n_users <= 0 || cfg.n_groups <= 0) {
    throw std::invalid_argument("UgConfig must have positive n_users and n_groups");
  }
  if (cfg.s_min <= 0 || cfg.s_max < cfg.s_min) {
    throw std::invalid_argument("UgConfig invalid s_min/s_max");
  }
  if (cfg.alpha <= 0.0) {
    throw std::invalid_argument("UgConfig alpha must be positive");
  }
  if (cfg.user_mean <= 0.0) {
    throw std::invalid_argument("UgConfig user_mean must be positive");
  }
  if (cfg.user_r_max <= 0) {
    throw std::invalid_argument("UgConfig user_r_max must be positive");
  }
}

std::vector<Index> SamplePowerLawSizes(const UgConfig& cfg,
                                       std::mt19937_64& rng) {
  // Sample group sizes from a truncated power-law distribution.
  const Index range = cfg.s_max - cfg.s_min + 1;
  if (range <= 0) {
    throw std::invalid_argument("Invalid power-law size range");
  }

  std::vector<double> cdf(static_cast<size_t>(range));
  double acc = 0.0;
  for (Index i = 0; i < range; ++i) {
    Index s = cfg.s_min + i;
    acc += std::pow(static_cast<double>(s), -cfg.alpha);
    cdf[static_cast<size_t>(i)] = acc;
  }
  if (acc <= 0.0 || !std::isfinite(acc)) {
    throw std::runtime_error("Power-law normalization invalid");
  }

  std::uniform_real_distribution<double> dist(0.0, acc);
  std::vector<Index> sizes(static_cast<size_t>(cfg.n_groups));
  for (Index g = 0; g < cfg.n_groups; ++g) {
    double u = dist(rng);
    auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
    Index idx = static_cast<Index>(it - cdf.begin());
    sizes[static_cast<size_t>(g)] = cfg.s_min + idx;
  }
  return sizes;
}

int64_t SumInt64(const std::vector<Index>& values) {
  // Sum integers using 64-bit accumulation.
  return std::accumulate(values.begin(), values.end(), int64_t{0});
}

void AdjustDegreeSum(std::vector<Index>& degrees, int64_t target_sum,
                     Index r_max, std::mt19937_64& rng) {
  // Adjust degrees to match a target sum while respecting bounds.
  int64_t sum = SumInt64(degrees);
  if (sum == target_sum) {
    return;
  }

  if (sum < target_sum) {
    std::vector<Index> candidates;
    candidates.reserve(degrees.size());
    for (Index i = 0; i < static_cast<Index>(degrees.size()); ++i) {
      if (degrees[static_cast<size_t>(i)] < r_max) {
        candidates.push_back(i);
      }
    }
    if (candidates.empty()) {
      throw std::runtime_error("Cannot increase degrees to target sum");
    }
    int64_t delta = target_sum - sum;
    while (delta > 0) {
      std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
      size_t pos = dist(rng);
      Index idx = candidates[pos];
      degrees[static_cast<size_t>(idx)] += 1;
      --delta;
      if (degrees[static_cast<size_t>(idx)] >= r_max) {
        candidates[pos] = candidates.back();
        candidates.pop_back();
        if (candidates.empty() && delta > 0) {
          throw std::runtime_error("Cannot increase degrees to target sum");
        }
      }
    }
    return;
  }

  std::vector<Index> candidates;
  candidates.reserve(degrees.size());
  for (Index i = 0; i < static_cast<Index>(degrees.size()); ++i) {
    if (degrees[static_cast<size_t>(i)] > 1) {
      candidates.push_back(i);
    }
  }
  if (candidates.empty()) {
    throw std::runtime_error("Cannot decrease degrees to target sum");
  }
  int64_t delta = sum - target_sum;
  while (delta > 0) {
    std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
    size_t pos = dist(rng);
    Index idx = candidates[pos];
    degrees[static_cast<size_t>(idx)] -= 1;
    --delta;
    if (degrees[static_cast<size_t>(idx)] <= 1) {
      candidates[pos] = candidates.back();
      candidates.pop_back();
      if (candidates.empty() && delta > 0) {
        throw std::runtime_error("Cannot decrease degrees to target sum");
      }
    }
  }
}

std::vector<Index> SampleUserDegrees(const UgConfig& cfg,
                                     std::mt19937_64& rng,
                                     int64_t target_sum) {
  // Sample user participation degrees and adjust to match total stubs.
  if (target_sum < cfg.n_users) {
    throw std::runtime_error("Target sum too small for min user degree 1");
  }
  if (static_cast<int64_t>(cfg.user_r_max) * cfg.n_users < target_sum) {
    throw std::runtime_error("Target sum exceeds maximum possible user degrees");
  }

  std::poisson_distribution<int> dist(cfg.user_mean);
  std::vector<Index> degrees(static_cast<size_t>(cfg.n_users));
  for (Index u = 0; u < cfg.n_users; ++u) {
    int val = dist(rng);
    if (val < 1) {
      val = 1;
    }
    if (val > cfg.user_r_max) {
      val = cfg.user_r_max;
    }
    degrees[static_cast<size_t>(u)] = static_cast<Index>(val);
  }

  AdjustDegreeSum(degrees, target_sum, cfg.user_r_max, rng);
  return degrees;
}

std::vector<Index> BuildStubs(const std::vector<Index>& counts, int64_t total) {
  // Expand counts into a list of stub indices.
  if (total < 0) {
    throw std::invalid_argument("Stub total cannot be negative");
  }
  if (total > std::numeric_limits<Index>::max()) {
    throw std::runtime_error("Stub total exceeds index limit");
  }
  std::vector<Index> stubs;
  stubs.reserve(static_cast<size_t>(total));
  for (Index i = 0; i < static_cast<Index>(counts.size()); ++i) {
    Index c = counts[static_cast<size_t>(i)];
    for (Index k = 0; k < c; ++k) {
      stubs.push_back(i);
    }
  }
  return stubs;
}

BipartiteCsr BuildBipartiteFromStubs(
    Index n_users,
    Index n_groups,
    std::vector<Index>& user_stubs,
    std::vector<Index>& group_stubs,
    const std::vector<Index>& user_degrees,
    std::mt19937_64& rng) {
  // Pair shuffled stubs and build CSR without a global pair list.
  if (user_stubs.size() != group_stubs.size()) {
    throw std::runtime_error("Stub sizes do not match");
  }
  std::shuffle(user_stubs.begin(), user_stubs.end(), rng);
  std::shuffle(group_stubs.begin(), group_stubs.end(), rng);

  std::vector<Index> user_offsets(static_cast<size_t>(n_users + 1), 0);
  for (Index u = 0; u < n_users; ++u) {
    user_offsets[static_cast<size_t>(u + 1)] =
        user_offsets[static_cast<size_t>(u)] + user_degrees[static_cast<size_t>(u)];
  }
  if (static_cast<size_t>(user_offsets.back()) != user_stubs.size()) {
    throw std::runtime_error("User degree sum does not match stub count");
  }

  std::vector<Index> group_ids(user_stubs.size());
  std::vector<Index> fill_offsets = user_offsets;
  for (size_t i = 0; i < user_stubs.size(); ++i) {
    const Index u = user_stubs[i];
    const Index pos = fill_offsets[static_cast<size_t>(u)]++;
    group_ids[static_cast<size_t>(pos)] = group_stubs[i];
  }

  std::vector<Index>().swap(user_stubs);
  std::vector<Index>().swap(group_stubs);

  std::vector<Index> row_counts(static_cast<size_t>(n_users), 0);
  for (Index u = 0; u < n_users; ++u) {
    const size_t begin = static_cast<size_t>(user_offsets[static_cast<size_t>(u)]);
    const size_t end = static_cast<size_t>(user_offsets[static_cast<size_t>(u + 1)]);
    if (begin == end) {
      continue;
    }
    auto first = group_ids.begin() + static_cast<std::ptrdiff_t>(begin);
    auto last = group_ids.begin() + static_cast<std::ptrdiff_t>(end);
    std::sort(first, last);
    Index unique = 1;
    for (auto it = first + 1; it != last; ++it) {
      if (*it != *(it - 1)) {
        unique += 1;
      }
    }
    row_counts[static_cast<size_t>(u)] = unique;
  }

  std::vector<Index> row_ptr(static_cast<size_t>(n_users + 1), 0);
  for (Index u = 0; u < n_users; ++u) {
    row_ptr[static_cast<size_t>(u + 1)] =
        row_ptr[static_cast<size_t>(u)] + row_counts[static_cast<size_t>(u)];
  }

  const Index nnz = row_ptr.back();
  std::vector<Index> col_idx(static_cast<size_t>(nnz));
  std::vector<Scalar> values(static_cast<size_t>(nnz));
  std::vector<Index> offsets = row_ptr;

  for (Index u = 0; u < n_users; ++u) {
    const size_t begin = static_cast<size_t>(user_offsets[static_cast<size_t>(u)]);
    const size_t end = static_cast<size_t>(user_offsets[static_cast<size_t>(u + 1)]);
    if (begin == end) {
      continue;
    }
    size_t i = begin;
    while (i < end) {
      size_t j = i + 1;
      while (j < end && group_ids[j] == group_ids[i]) {
        ++j;
      }
      const Index pos = offsets[static_cast<size_t>(u)]++;
      col_idx[static_cast<size_t>(pos)] = group_ids[i];
      values[static_cast<size_t>(pos)] = static_cast<double>(j - i);
      i = j;
    }
  }

  return BipartiteCsr::FromCsr(n_users, n_groups, std::move(row_ptr),
                               std::move(col_idx), std::move(values));
}

Vector ToVector(const std::vector<Index>& values) {
  // Convert integer vector to dense double vector.
  Vector v(static_cast<Index>(values.size()));
  for (Index i = 0; i < static_cast<Index>(values.size()); ++i) {
    v[i] = static_cast<double>(values[static_cast<size_t>(i)]);
  }
  return v;
}

}  // namespace

UgResult UgGenerator::Generate(const UgConfig& cfg) {
  // Generate a user-group bipartite graph with power-law group sizes.
  ValidateConfig(cfg);

  std::mt19937_64 rng(cfg.seed);

  std::vector<Index> group_sizes = SamplePowerLawSizes(cfg, rng);
  int64_t total_stubs = SumInt64(group_sizes);
  if (total_stubs <= 0) {
    throw std::runtime_error("Generated empty group sizes");
  }
  if (total_stubs > std::numeric_limits<Index>::max()) {
    throw std::runtime_error("Total stubs exceed index limit");
  }

  std::vector<Index> user_degrees =
      SampleUserDegrees(cfg, rng, total_stubs);

  std::vector<Index> user_stubs = BuildStubs(user_degrees, total_stubs);
  std::vector<Index> group_stubs = BuildStubs(group_sizes, total_stubs);

  UgResult result;
  result.bipartite = BuildBipartiteFromStubs(cfg.n_users, cfg.n_groups,
                                             user_stubs, group_stubs,
                                             user_degrees, rng);
  result.group_sizes = ToVector(group_sizes);
  result.user_degrees = ToVector(user_degrees);
  return result;
}

}  // namespace fj
