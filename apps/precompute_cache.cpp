#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include "fj/experiment/experiment_instance.hpp"
#include "fj/graph/bipartite_csr.hpp"
#include "fj/graph/csr_graph.hpp"
#include "fj/io/binary_csr_io.hpp"
#include "fj/io/edge_list_reader.hpp"
#include "fj/io/precond_io.hpp"
#include "fj/preconditioner/precond_builder.hpp"

namespace {

void PrintUsage() {
  // Print CLI usage for the preprocessing tool.
  std::cout << "Usage: precompute_cache [options]\n"
            << "Options (use --key value or --key=value):\n"
            << "  --bipartite              (string) user-group edge list path\n"
            << "  --user_graph             (string) user graph edge list path (optional)\n"
            << "  --group_graph            (string) group graph edge list path (optional)\n"
            << "  --out_bipartite_bin       (string) output binary CSR path for UG\n"
            << "  --out_user_graph_bin      (string) output binary CSR path for user graph\n"
            << "  --out_group_graph_bin     (string) output binary CSR path for group graph\n"
            << "  --out_schur_precond       (string) output Schur Jacobi preconditioner\n"
            << "  --out_full_precond        (string) output full-system Jacobi preconditioner\n"
            << "  --n_users                 (int)    number of users (optional)\n"
            << "  --n_groups                (int)    number of groups (optional)\n"
            << "  --data_one_indexed        (bool)   input edges are 1-indexed\n"
            << "  --data_ignore_self_loops  (bool)   ignore self loops in inputs\n"
            << "  --user_symmetrize         (bool)   symmetrize user graph edges\n"
            << "  --group_symmetrize        (bool)   symmetrize group graph edges\n"
            << "  --lambda_user             (double) user anchoring for preconditioners\n"
            << "  --lambda_group            (double) group anchoring for preconditioners\n"
            << "  -h, --help                          show this help\n";
}

bool ParseBool(const std::string& s, bool& out) {
  if (s == "1" || s == "true" || s == "yes" || s == "on") {
    out = true;
    return true;
  }
  if (s == "0" || s == "false" || s == "no" || s == "off") {
    out = false;
    return true;
  }
  return false;
}

bool ConsumeArg(int argc, char** argv, int& i, std::string& key,
                std::string& value) {
  std::string arg = argv[i];
  if (arg == "-h" || arg == "--help") {
    key = "help";
    return true;
  }
  if (arg.rfind("--", 0) != 0) {
    return false;
  }
  arg = arg.substr(2);
  size_t eq = arg.find('=');
  if (eq != std::string::npos) {
    key = arg.substr(0, eq);
    value = arg.substr(eq + 1);
    return true;
  }
  key = arg;
  if (i + 1 >= argc) {
    throw std::invalid_argument("Missing value for --" + key);
  }
  value = argv[++i];
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  fj::Index n_users = 0;
  fj::Index n_groups = 0;
  bool n_users_set = false;
  bool n_groups_set = false;

  std::string bipartite_path;
  std::string user_graph_path;
  std::string group_graph_path;
  std::string out_bipartite_bin;
  std::string out_user_graph_bin;
  std::string out_group_graph_bin;
  std::string out_schur_precond;
  std::string out_full_precond;

  bool data_one_indexed = false;
  bool data_ignore_self_loops = true;
  bool user_symmetrize = false;
  bool group_symmetrize = false;

  double lambda_user = 1.0;
  double lambda_group = 1e-4;

  try {
    for (int i = 1; i < argc; ++i) {
      std::string key;
      std::string value;
      if (!ConsumeArg(argc, argv, i, key, value)) {
        throw std::invalid_argument("Unknown argument: " + std::string(argv[i]));
      }
      if (key == "help") {
        PrintUsage();
        return 0;
      } else if (key == "bipartite") {
        bipartite_path = value;
      } else if (key == "user_graph") {
        user_graph_path = value;
      } else if (key == "group_graph") {
        group_graph_path = value;
      } else if (key == "out_bipartite_bin") {
        out_bipartite_bin = value;
      } else if (key == "out_user_graph_bin") {
        out_user_graph_bin = value;
      } else if (key == "out_group_graph_bin") {
        out_group_graph_bin = value;
      } else if (key == "out_schur_precond") {
        out_schur_precond = value;
      } else if (key == "out_full_precond") {
        out_full_precond = value;
      } else if (key == "n_users") {
        n_users = static_cast<fj::Index>(std::stoll(value));
        n_users_set = true;
      } else if (key == "n_groups") {
        n_groups = static_cast<fj::Index>(std::stoll(value));
        n_groups_set = true;
      } else if (key == "data_one_indexed") {
        if (!ParseBool(value, data_one_indexed)) {
          throw std::invalid_argument("Invalid bool for data_one_indexed");
        }
      } else if (key == "data_ignore_self_loops") {
        if (!ParseBool(value, data_ignore_self_loops)) {
          throw std::invalid_argument("Invalid bool for data_ignore_self_loops");
        }
      } else if (key == "user_symmetrize") {
        if (!ParseBool(value, user_symmetrize)) {
          throw std::invalid_argument("Invalid bool for user_symmetrize");
        }
      } else if (key == "group_symmetrize") {
        if (!ParseBool(value, group_symmetrize)) {
          throw std::invalid_argument("Invalid bool for group_symmetrize");
        }
      } else if (key == "lambda_user") {
        lambda_user = std::stod(value);
      } else if (key == "lambda_group") {
        lambda_group = std::stod(value);
      } else {
        throw std::invalid_argument("Unknown option: --" + key);
      }
    }
  } catch (const std::exception& ex) {
    std::cerr << "Argument error: " << ex.what() << "\n";
    PrintUsage();
    return 2;
  }

  if (bipartite_path.empty()) {
    std::cerr << "Error: --bipartite is required\n";
    PrintUsage();
    return 2;
  }
  if (lambda_user <= 0.0 || lambda_group < 0.0) {
    std::cerr << "Error: lambda_user must be positive and lambda_group nonnegative\n";
    return 2;
  }

  if (!n_users_set) {
    n_users = 0;
  }
  if (!n_groups_set) {
    n_groups = 0;
  }

  fj::EdgeListOptions ug_opts;
  ug_opts.one_indexed = data_one_indexed;
  ug_opts.ignore_self_loops = data_ignore_self_loops;
  ug_opts.symmetrize = false;

  fj::BipartiteCsr bipartite =
      fj::EdgeListReader::ReadBipartite(bipartite_path, n_users, n_groups, ug_opts);

  const fj::Index resolved_users = bipartite.num_users();
  const fj::Index resolved_groups = bipartite.num_groups();

  fj::EdgeListOptions user_opts = ug_opts;
  user_opts.symmetrize = user_symmetrize;
  fj::WeightedCsrGraph user_graph =
      user_graph_path.empty()
          ? fj::WeightedCsrGraph(resolved_users)
          : fj::EdgeListReader::ReadWeightedGraph(user_graph_path, resolved_users,
                                                  user_opts);

  fj::EdgeListOptions group_opts = ug_opts;
  group_opts.symmetrize = group_symmetrize;
  fj::WeightedCsrGraph group_graph =
      group_graph_path.empty()
          ? fj::WeightedCsrGraph(resolved_groups)
          : fj::EdgeListReader::ReadWeightedGraph(group_graph_path, resolved_groups,
                                                  group_opts);

  if (!out_bipartite_bin.empty()) {
    fj::BinaryCsrIO::WriteBipartite(out_bipartite_bin, bipartite);
  }
  if (!out_user_graph_bin.empty()) {
    fj::BinaryCsrIO::WriteWeightedGraph(out_user_graph_bin, user_graph);
  }
  if (!out_group_graph_bin.empty()) {
    fj::BinaryCsrIO::WriteWeightedGraph(out_group_graph_bin, group_graph);
  }

  if (!out_schur_precond.empty() || !out_full_precond.empty()) {
    fj::ExperimentInstance instance;
    instance.user_graph = std::move(user_graph);
    instance.group_graph = std::move(group_graph);
    instance.bipartite = std::move(bipartite);
    instance.lambda_u = fj::Vector::Constant(resolved_users, lambda_user);
    instance.lambda_g = fj::Vector::Constant(resolved_groups, lambda_group);

    if (!out_schur_precond.empty()) {
      fj::Vector diag = fj::BuildSchurJacobiDiagonal(instance);
      fj::PrecondIO::WriteJacobiDiag(out_schur_precond,
                                     fj::PrecondKind::kSchurJacobi, diag,
                                     resolved_users, resolved_groups,
                                     lambda_user, lambda_group);
    }
    if (!out_full_precond.empty()) {
      fj::Vector diag = fj::BuildFullJacobiDiagonal(instance);
      fj::PrecondIO::WriteJacobiDiag(out_full_precond,
                                     fj::PrecondKind::kFullJacobi, diag,
                                     resolved_users, resolved_groups,
                                     lambda_user, lambda_group);
    }
  }

  return 0;
}
