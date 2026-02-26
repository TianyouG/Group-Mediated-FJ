#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "fj/experiment/experiment_runner.hpp"
#include "fj/experiment/method.hpp"
#include "fj/io/csv_writer.hpp"

namespace {

void PrintUsage() {
  // Print CLI usage and all supported options.
  std::cout << "Usage: run_experiments [options]\\n"
            << "Options (use --key value or --key=value):\\n"
            << "  --n_users            (int)    number of users\\n"
            << "  --n_groups           (int)    number of groups\\n"
            << "  --method             (string) schur|full_system|clique|direct|fj_dynamics\\n"
            << "  --input_mode         (string) synthetic|data\\n"
            << "  --user_graph         (string) user graph edge list path\\n"
            << "  --group_graph        (string) group graph edge list path (optional)\\n"
            << "  --bipartite          (string) user-group edge list path\\n"
            << "  --user_graph_bin     (string) user graph binary CSR path\\n"
            << "  --group_graph_bin    (string) group graph binary CSR path\\n"
            << "  --bipartite_bin      (string) user-group binary CSR path\\n"
            << "  --su                 (string) user prior vector file (data mode)\\n"
            << "  --sg                 (string) group prior vector file (data mode)\\n"
            << "  --lambda_u           (string) user lambda vector file (data mode)\\n"
            << "  --lambda_g           (string) group lambda vector file (data mode)\\n"
            << "  --data_one_indexed   (bool)   input edges are 1-indexed\\n"
            << "  --data_ignore_self_loops (bool) ignore self loops in inputs\\n"
            << "  --user_symmetrize    (bool)   symmetrize user graph edges\\n"
            << "  --group_symmetrize   (bool)   symmetrize group graph edges\\n"
            << "  --alpha              (double) power-law exponent\\n"
            << "  --s_min              (int)    min group size\\n"
            << "  --s_max              (int)    max group size\\n"
            << "  --user_mean          (double) mean user degree (Poisson)\\n"
            << "  --user_r_max         (int)    max user degree\\n"
            << "  --ug_normalize_groups (bool)  normalize UG weights per group (synthetic)\\n"
            << "  --su_distribution    (string) uniform|exponential|powerlaw\\n"
            << "  --exp_lambda         (double) exponential lambda for synthetic s_u\\n"
            << "  --power_alpha        (double) power-law alpha for synthetic s_u\\n"
            << "  --power_min          (double) power-law min for synthetic s_u\\n"
            << "  --power_max          (double) power-law max for synthetic s_u\\n"
            << "  --k_ws               (int)    WS lattice degree (even)\\n"
            << "  --p_ws               (double) WS rewiring probability\\n"
            << "  --user_graph_weight  (double) synthetic user graph weight (-1 auto)\\n"
            << "  --enable_group_graph (bool)   enable group graph\\n"
            << "  --group_overlap_threshold (double) min overlap ratio for group edges\\n"
            << "  --group_graph_target_mean_degree (double) target mean weighted degree of group graph\\n"
            << "  --lambda_user        (double) user anchoring\\n"
            << "  --lambda_group       (double) group anchoring\\n"
            << "  --outer_max_iters    (int)    outer CG max iters\\n"
            << "  --outer_tol          (double) outer CG tolerance\\n"
            << "  --inner_max_iters    (int)    inner CG max iters\\n"
            << "  --inner_tol          (double) inner CG tolerance\\n"
            << "  --clique_use_weights (bool)   clique baseline uses weights\\n"
            << "  --schur_jacobi       (bool)   schur uses Jacobi precond\\n"
            << "  --full_system_jacobi (bool)   full_system uses Jacobi precond\\n"
            << "  --schur_precond      (string) schur Jacobi preconditioner file\\n"
            << "  --full_precond       (string) full-system Jacobi preconditioner file\\n"
            << "  --direct_max_dim     (int)    max dimension for direct solve\\n"
            << "  --seed               (int)    RNG seed\\n"
            << "  --tag                (string) experiment tag\\n"
            << "  --csv                (string) output CSV path\\n"
            << "  -h, --help                     show this help\\n";
}

bool ParseBool(const std::string& s, bool& out) {
  // Parse common boolean tokens into a bool.
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
  // Parse a --key value or --key=value argument; supports -h/--help.
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

std::string ToString(double v) {
  // Convert double to string with full precision for CSV output.
  std::ostringstream os;
  os.precision(17);
  os << v;
  return os.str();
}

void WriteCsv(const fj::ExperimentConfig& cfg, const fj::ExperimentResult& res,
              const std::string& path) {
  // Serialize config + result into a key/value CSV file.
  fj::CsvWriter writer(path);
  writer.WriteHeader({"key", "value"});

  auto write = [&writer](const std::string& key, const std::string& value) {
    writer.WriteRow({key, value});
  };

  write("tag", res.tag);
  write("method", fj::ToString(res.method));
  write("n_users", std::to_string(cfg.n_users));
  write("n_groups", std::to_string(cfg.n_groups));
  write("alpha", ToString(cfg.alpha));
  write("s_min", std::to_string(cfg.s_min));
  write("s_max", std::to_string(cfg.s_max));
  write("user_mean", ToString(cfg.user_mean));
  write("user_r_max", std::to_string(cfg.user_r_max));
  write("ug_normalize_groups", cfg.ug_normalize_groups ? "1" : "0");
  write("su_distribution", cfg.su_distribution);
  write("exp_lambda", ToString(cfg.exp_lambda));
  write("power_alpha", ToString(cfg.power_alpha));
  write("power_min", ToString(cfg.power_min));
  write("power_max", ToString(cfg.power_max));
  write("k_ws", std::to_string(cfg.k_ws));
  write("p_ws", ToString(cfg.p_ws));
  write("user_graph_weight", ToString(res.user_graph_weight));
  write("enable_group_graph", cfg.enable_group_graph ? "1" : "0");
  write("group_overlap_threshold", ToString(cfg.group_overlap_threshold));
  write("group_graph_target_mean_degree",
        ToString(cfg.group_graph_target_mean_degree));
  write("lambda_user", ToString(cfg.lambda_user));
  write("lambda_group", ToString(cfg.lambda_group));
  write("outer_max_iters", std::to_string(cfg.outer_max_iters));
  write("outer_tol", ToString(cfg.outer_tol));
  write("inner_max_iters", std::to_string(cfg.inner_max_iters));
  write("inner_tol", ToString(cfg.inner_tol));
  write("clique_use_weights", cfg.clique_use_weights ? "1" : "0");
  write("schur_jacobi", cfg.schur_use_jacobi ? "1" : "0");
  write("full_system_jacobi", cfg.full_system_use_jacobi ? "1" : "0");
  write("direct_max_dim", std::to_string(cfg.direct_max_dim));
  write("seed", std::to_string(cfg.seed));
  write("outer_iters", std::to_string(res.outer_iters));
  write("inner_iters", std::to_string(res.inner_iters));
  write("inner_seconds", ToString(res.inner_seconds));
  write("seconds", ToString(res.seconds));
  write("relative_residual", ToString(res.relative_residual));
  write("disagreement", ToString(res.disagreement));
  write("internal_conflict", ToString(res.internal_conflict));
  write("polarization", ToString(res.polarization));
  write("controversy", ToString(res.controversy));
  write("nnz_w", std::to_string(res.nnz_w));
  write("max_s_g", ToString(res.max_group_size));
  write("sum_s_g_sq", ToString(res.sum_group_size_sq));
  write("user_cross_deg_mean", ToString(res.user_cross_degree_mean));
  write("user_cross_deg_max", ToString(res.user_cross_degree_max));
  write("group_cross_deg_mean", ToString(res.group_cross_degree_mean));
  write("group_cross_deg_max", ToString(res.group_cross_degree_max));
  write("user_graph_deg_mean", ToString(res.user_graph_degree_mean));
  write("user_graph_deg_max", ToString(res.user_graph_degree_max));
  write("group_graph_deg_mean", ToString(res.group_graph_degree_mean));
  write("group_graph_deg_max", ToString(res.group_graph_degree_max));
}

}  // namespace

int main(int argc, char** argv) {
  // Entry point: parse CLI, run one experiment, optionally write CSV.
  fj::ExperimentConfig cfg;
  cfg.n_users = 100;
  cfg.n_groups = 50;
  cfg.method = fj::Method::Schur;
  cfg.alpha = 2.2;
  cfg.s_min = 1;
  cfg.s_max = 50;
  cfg.user_mean = 3.0;
  cfg.user_r_max = 20;
  cfg.ug_normalize_groups = true;
  cfg.su_distribution = "uniform";
  cfg.exp_lambda = 1.0;
  cfg.power_alpha = 2.0;
  cfg.power_min = 1.0;
  cfg.power_max = 100.0;
  cfg.k_ws = 6;
  cfg.p_ws = 0.1;
  cfg.user_graph_weight = -1.0;
  cfg.enable_group_graph = false;
  cfg.group_overlap_threshold = 0.5;
  cfg.group_graph_target_mean_degree = 1.0;
  cfg.lambda_user = 1.0;
  cfg.lambda_group = 1e-4;
  cfg.outer_max_iters = 200;
  cfg.outer_tol = 1e-8;
  cfg.inner_max_iters = 200;
  cfg.inner_tol = 1e-8;
  cfg.clique_use_weights = false;
  cfg.schur_use_jacobi = true;
  cfg.full_system_use_jacobi = true;
  cfg.direct_max_dim = 50000;
  cfg.seed = 1;
  cfg.tag = "default";

  bool n_users_set = false;
  bool n_groups_set = false;
  std::string csv_path;

  try {
    // Parse CLI arguments into config.
    for (int i = 1; i < argc; ++i) {
      std::string key;
      std::string value;
      if (!ConsumeArg(argc, argv, i, key, value)) {
        throw std::invalid_argument("Unknown argument: " + std::string(argv[i]));
      }
      if (key == "help") {
        PrintUsage();
        return 0;
      } else if (key == "method") {
        cfg.method = fj::ParseMethod(value);
      } else if (key == "input_mode") {
        if (value == "synthetic") {
          cfg.use_real_data = false;
        } else if (value == "data") {
          cfg.use_real_data = true;
        } else {
          throw std::invalid_argument("Unknown input_mode: " + value);
        }
      } else if (key == "user_graph") {
        cfg.user_graph_path = value;
      } else if (key == "group_graph") {
        cfg.group_graph_path = value;
      } else if (key == "bipartite") {
        cfg.bipartite_path = value;
      } else if (key == "user_graph_bin") {
        cfg.user_graph_bin_path = value;
      } else if (key == "group_graph_bin") {
        cfg.group_graph_bin_path = value;
      } else if (key == "bipartite_bin") {
        cfg.bipartite_bin_path = value;
      } else if (key == "su") {
        cfg.su_path = value;
      } else if (key == "sg") {
        cfg.sg_path = value;
      } else if (key == "lambda_u") {
        cfg.lambda_u_path = value;
      } else if (key == "lambda_g") {
        cfg.lambda_g_path = value;
      } else if (key == "data_one_indexed") {
        if (!ParseBool(value, cfg.data_one_indexed)) {
          throw std::invalid_argument("Invalid bool for data_one_indexed");
        }
      } else if (key == "data_ignore_self_loops") {
        if (!ParseBool(value, cfg.data_ignore_self_loops)) {
          throw std::invalid_argument("Invalid bool for data_ignore_self_loops");
        }
      } else if (key == "user_symmetrize") {
        if (!ParseBool(value, cfg.user_graph_symmetrize)) {
          throw std::invalid_argument("Invalid bool for user_symmetrize");
        }
      } else if (key == "group_symmetrize") {
        if (!ParseBool(value, cfg.group_graph_symmetrize)) {
          throw std::invalid_argument("Invalid bool for group_symmetrize");
        }
      } else if (key == "n_users") {
        cfg.n_users = static_cast<fj::Index>(std::stoll(value));
        n_users_set = true;
      } else if (key == "n_groups") {
        cfg.n_groups = static_cast<fj::Index>(std::stoll(value));
        n_groups_set = true;
      } else if (key == "alpha") {
        cfg.alpha = std::stod(value);
      } else if (key == "s_min") {
        cfg.s_min = static_cast<fj::Index>(std::stoll(value));
      } else if (key == "s_max") {
        cfg.s_max = static_cast<fj::Index>(std::stoll(value));
      } else if (key == "user_mean") {
        cfg.user_mean = std::stod(value);
      } else if (key == "user_r_max") {
        cfg.user_r_max = static_cast<fj::Index>(std::stoll(value));
      } else if (key == "ug_normalize_groups") {
        if (!ParseBool(value, cfg.ug_normalize_groups)) {
          throw std::invalid_argument("Invalid bool for ug_normalize_groups");
        }
      } else if (key == "su_distribution") {
        cfg.su_distribution = value;
      } else if (key == "exp_lambda") {
        cfg.exp_lambda = std::stod(value);
      } else if (key == "power_alpha") {
        cfg.power_alpha = std::stod(value);
      } else if (key == "power_min") {
        cfg.power_min = std::stod(value);
      } else if (key == "power_max") {
        cfg.power_max = std::stod(value);
      } else if (key == "k_ws") {
        cfg.k_ws = static_cast<fj::Index>(std::stoll(value));
      } else if (key == "p_ws") {
        cfg.p_ws = std::stod(value);
      } else if (key == "user_graph_weight") {
        cfg.user_graph_weight = std::stod(value);
      } else if (key == "enable_group_graph") {
        if (!ParseBool(value, cfg.enable_group_graph)) {
          throw std::invalid_argument("Invalid bool for enable_group_graph");
        }
      } else if (key == "group_overlap_threshold") {
        cfg.group_overlap_threshold = std::stod(value);
      } else if (key == "group_graph_target_mean_degree") {
        cfg.group_graph_target_mean_degree = std::stod(value);
      } else if (key == "lambda_user") {
        cfg.lambda_user = std::stod(value);
      } else if (key == "lambda_group") {
        cfg.lambda_group = std::stod(value);
      } else if (key == "outer_max_iters") {
        cfg.outer_max_iters = static_cast<fj::Index>(std::stoll(value));
      } else if (key == "outer_tol") {
        cfg.outer_tol = std::stod(value);
      } else if (key == "inner_max_iters") {
        cfg.inner_max_iters = static_cast<fj::Index>(std::stoll(value));
      } else if (key == "inner_tol") {
        cfg.inner_tol = std::stod(value);
      } else if (key == "clique_use_weights") {
        bool flag = false;
        if (!ParseBool(value, flag)) {
          throw std::invalid_argument("Invalid bool for clique_use_weights");
        }
        cfg.clique_use_weights = flag;
      } else if (key == "schur_jacobi") {
        bool flag = false;
        if (!ParseBool(value, flag)) {
          throw std::invalid_argument("Invalid bool for schur_jacobi");
        }
        cfg.schur_use_jacobi = flag;
      } else if (key == "full_system_jacobi") {
        bool flag = false;
        if (!ParseBool(value, flag)) {
          throw std::invalid_argument("Invalid bool for full_system_jacobi");
        }
        cfg.full_system_use_jacobi = flag;
      } else if (key == "schur_precond") {
        cfg.schur_precond_path = value;
      } else if (key == "full_precond") {
        cfg.full_precond_path = value;
      } else if (key == "direct_max_dim") {
        cfg.direct_max_dim = static_cast<fj::Index>(std::stoll(value));
      } else if (key == "seed") {
        cfg.seed = static_cast<uint64_t>(std::stoull(value));
      } else if (key == "tag") {
        cfg.tag = value;
      } else if (key == "csv") {
        csv_path = value;
      } else {
        throw std::invalid_argument("Unknown option: --" + key);
      }
    }
  } catch (const std::exception& ex) {
    std::cerr << "Argument error: " << ex.what() << "\n";
    PrintUsage();
    return 2;
  }

  if (cfg.use_real_data) {
    // Allow inferring sizes from the bipartite file when not specified.
    if (!n_users_set) {
      cfg.n_users = 0;
    }
    if (!n_groups_set) {
      cfg.n_groups = 0;
    }
  } else if (!cfg.su_path.empty() || !cfg.sg_path.empty() ||
             !cfg.lambda_u_path.empty() || !cfg.lambda_g_path.empty()) {
    std::cerr << "Error: s/lambda vector files are only supported with "
                 "--input_mode data\n";
    return 2;
  }

  try {
    // Build instance, run method, and emit outputs.
    fj::ExperimentRunner runner(cfg);
    fj::ExperimentResult result = runner.Run();

    std::cout << "method=" << fj::ToString(result.method) << "\n";
    std::cout << "tag=" << result.tag << "\n";
    std::cout << "outer_iters=" << result.outer_iters << "\n";
    std::cout << "inner_iters=" << result.inner_iters << "\n";
    std::cout << "seconds=" << result.seconds << "\n";
    std::cout << "relative_residual=" << result.relative_residual << "\n";
    std::cout << "disagreement=" << result.disagreement << "\n";
    std::cout << "internal_conflict=" << result.internal_conflict << "\n";
    std::cout << "polarization=" << result.polarization << "\n";
    std::cout << "controversy=" << result.controversy << "\n";

    if (!csv_path.empty()) {
      WriteCsv(cfg, result, csv_path);
    }
  } catch (const std::exception& ex) {
    std::cerr << "Run failed: " << ex.what() << "\n";
    return 1;
  }

  return 0;
}
