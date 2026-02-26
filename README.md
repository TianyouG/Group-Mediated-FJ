# The two-layer FJ model

This repository provides a C++17 implementation for computing the steady state of a two-layer Friedkin--Johnsen (FJ) model on:
- a user graph,
- a group graph,
- and a user-group bipartite graph.

## Requirements

- C++17 compiler (`g++ >= 9` recommended)
- CMake (`>= 3.16`)

## Build

```bash
cmake -S . -B build_o3 -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG"
cmake --build build_o3 -j
```

Main executables:

- `build_o3/run_experiments`
- `build_o3/precompute_cache`

## Quick Start 1: Synthetic Graph

```bash
./build_o3/run_experiments \
  --input_mode synthetic \
  --method schur \
  --schur_jacobi true \
  --n_users 1000000 \
  --n_groups 500000 \
  --seed 1 \
  --csv out.csv
```

## Quick Start 2: Real Data (Edge Lists)

```bash
./build_o3/run_experiments \
  --input_mode data \
  --method schur \
  --schur_jacobi true \
  --n_users 10000 \
  --n_groups 5000 \
  --bipartite data/ug.txt \
  --user_graph data/user_graph.txt \
  --group_graph data/group_graph.txt \
  --su data/su.txt \
  --sg data/sg.txt \
  --data_one_indexed false \
  --user_symmetrize true \
  --group_symmetrize true \
  --csv out.csv
```

## Key Parameters

Defaults below match `apps/run_experiments.cpp`.

| Parameter | Default | Notes |
|---|---:|---|
| `--method` | `schur` | `schur`, `full_system`, `clique`, `fj_dynamics`, `direct` |
| `--input_mode` | `synthetic` | `synthetic` or `data` |
| `--n_users` | `100` | Synthetic; in data mode omit to infer from `--bipartite` |
| `--n_groups` | `50` | Synthetic; in data mode omit to infer from `--bipartite` |
| `--bipartite` | *(empty)* | Required in data mode |
| `--user_graph` | *(empty)* | Optional in data mode (empty graph if omitted) |
| `--group_graph` | *(empty)* | Optional in data mode (empty graph if omitted) |
| `--su` | *(empty)* | Data mode: if empty, samples random +/-1 user priors |
| `--sg` | *(empty)* | Data mode: if empty, uses all-zero group priors |
| `--lambda_user` | `1.0` | Scalar user anchoring |
| `--lambda_group` | `1e-4` | Scalar group anchoring |
| `--outer_max_iters` | `200` | Outer iterations limit |
| `--outer_tol` | `1e-8` | Outer tolerance |
| `--inner_max_iters` | `200` | Inner iterations limit (mainly `schur`) |
| `--inner_tol` | `1e-8` | Inner tolerance (mainly `schur`) |
| `--schur_jacobi` | `true` | Enable Jacobi preconditioning for `schur` |
| `--full_system_jacobi` | `true` | Enable Jacobi preconditioning for `full_system` |
| `--csv` | *(empty)* | If empty, prints to stdout only |

## Input Format (Data Mode)

- Graph files (`user_graph`, `group_graph`, `bipartite`): one line per edge
  - `u v` or `u v w`
- Vector files (`su`, `sg`, `lambda_u`, `lambda_g`): one value per line
- Lines starting with `#` or `%` are ignored

## Help

```bash
./build_o3/run_experiments --help
./build_o3/precompute_cache --help
```
