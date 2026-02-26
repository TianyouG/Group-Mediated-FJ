#include "fj/model/fj_problem.hpp"

namespace fj {

// Compute the user-side RHS vector.
Vector FjProblem::b_u() const { return lambda_u.cwiseProduct(s_u); }

// Compute the group-side RHS vector.
Vector FjProblem::b_g() const { return lambda_g.cwiseProduct(s_g); }

}  // namespace fj
