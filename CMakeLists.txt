cmake_minimum_required(VERSION 3.16)
project(fj_two_layer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(fjcore
  src/baseline/clique_expansion.cpp
  src/baseline/direct_solver.cpp
  src/common/random.cpp
  src/experiment/experiment_runner.cpp
  src/experiment/method.cpp
  src/experiment/method_schur.cpp
  src/experiment/method_full_system.cpp
  src/experiment/method_clique.cpp
  src/experiment/method_direct.cpp
  src/experiment/method_fj_dynamics.cpp
  src/experiment/residuals.cpp
  src/generator/group_graph_generator.cpp
  src/generator/ug_generator.cpp
  src/generator/ws_generator.cpp
  src/graph/csr_graph.cpp
  src/graph/bipartite_csr.cpp
  src/metrics/metrics.cpp
  src/model/fj_problem.cpp
  src/operators/auu_operator.cpp
  src/operators/agg_operator.cpp
  src/operators/full_system_operator.cpp
  src/operators/schur_operator.cpp
  src/preconditioner/jacobi_preconditioner.cpp
  src/preconditioner/precond_builder.cpp
  src/solver/cg_solver.cpp
  src/solver/inner_solver.cpp
  src/io/csv_writer.cpp
  src/io/binary_csr_io.cpp
  src/io/precond_io.cpp
  src/io/edge_list_reader.cpp
  src/io/vector_reader.cpp
)

target_include_directories(fjcore PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/eigen
)

add_executable(run_experiments apps/run_experiments.cpp)

target_link_libraries(run_experiments PRIVATE fjcore)

add_executable(precompute_cache apps/precompute_cache.cpp)

target_link_libraries(precompute_cache PRIVATE fjcore)
