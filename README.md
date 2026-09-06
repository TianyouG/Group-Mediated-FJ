# Fast Computation of the Friedkin--Johnsen Steady State in Large-Scale Social Networks with Group-Mediated Interactions

This repository provides a C++17 implementation for computing the steady state
of a two-layer Friedkin--Johnsen model consisting of a user graph, a group graph,
and a user--group bipartite graph.

## Requirements

- A C++17 compiler (`g++ >= 9` recommended)
- CMake `>= 3.16`

Eigen is included and requires no separate installation.

## Build with `-O3`

```bash
cmake -S . -B build_o3 -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG"
cmake --build build_o3 -j
```

The main executable is `build_o3/run_experiments`.

## Quick Start: Synthetic Data

```bash
mkdir -p result

./build_o3/run_experiments \
  --input_mode synthetic \
  --method schur \
  --schur_jacobi true \
  --n_users 100000 \
  --n_groups 10000 \
  --seed 1 \
  --csv result/synthetic.csv
```

## Quick Start: Real-World Data

The following example uses the processed DBLP files under `data/dblp/power`:

```bash
mkdir -p result

./build_o3/run_experiments \
  --input_mode data \
  --method schur \
  --schur_jacobi true \
  --n_users 317080 \
  --n_groups 5136 \
  --user_graph data/dblp/power/user_graph.txt \
  --group_graph data/dblp/power/group_graph.txt \
  --bipartite data/dblp/power/ug.txt \
  --su data/dblp/power/su.txt \
  --sg data/dblp/power/sg.txt \
  --data_one_indexed false \
  --data_ignore_self_loops true \
  --user_symmetrize true \
  --group_symmetrize true \
  --lambda_user 1.0 \
  --lambda_group 0.1 \
  --outer_max_iters 1000 \
  --outer_tol 1e-4 \
  --inner_max_iters 1000 \
  --inner_tol 1e-4 \
  --seed 1 \
  --csv result/dblp_schur_pcg.csv
```

## Input Format

Graph files contain one edge per line:

```text
source destination
source destination weight
```

`user_graph.txt` and `group_graph.txt` contain within-layer edges. `ug.txt`
contains `user_id group_id weight`. Vector files such as `su.txt` and `sg.txt`
contain one value per line. Lines beginning with `#` or `%` are ignored.

## Key Parameters

| Parameter | Default | Description |
|---|---:|---|
| `--input_mode` | `synthetic` | `synthetic` or `data` |
| `--method` | `schur` | `schur`, `full_system`, `clique`, `fj_dynamics`, `direct`, `bli`, `bli_sor`, or `pf_qe` |
| `--n_users` | `100` | Number of user nodes |
| `--n_groups` | `50` | Number of group nodes |
| `--lambda_user` | `1.0` | User anchoring strength |
| `--lambda_group` | `1e-4` | Group anchoring strength |
| `--user_graph_scale` | `1.0` | User-graph edge-weight scaling factor |
| `--group_graph_scale` | `1.0` | Group-graph edge-weight scaling factor |
| `--outer_max_iters` | `200` | Maximum outer iterations |
| `--outer_tol` | `1e-8` | Outer relative-residual tolerance |
| `--inner_max_iters` | `200` | Maximum inner iterations for Schur |
| `--inner_tol` | `1e-8` | Inner relative-residual tolerance |
| `--schur_jacobi` | `true` | Enable the Schur Jacobi preconditioner |
| `--full_system_jacobi` | `true` | Enable the full-system Jacobi preconditioner |
| `--seed` | `1` | Random seed |
| `--csv` | empty | Output result file |

For all available parameters:

```bash
./build_o3/run_experiments --help
```
