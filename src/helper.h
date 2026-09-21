#ifndef EVBLM_HELPER_H
#define EVBLM_HELPER_H

#include <RcppArmadillo.h>
#include <functional>
#include <string>
#include <vector>

namespace evblm {

struct Evaluation {
  double value;
  arma::vec gradient;
};

struct OptimResult {
  arma::vec par;
  double value;
  int code;
  int fncount;
  int grcount;
  std::string message;
};

using Objective = std::function<Evaluation(const arma::vec&)>;
OptimResult optimize_lbfgsb(const arma::vec& initial, const arma::vec& lower,
                           const arma::vec& upper, const Objective& objective,
                           int maxit = 200, double factr = 100,
                           double pgtol = 1e-7);

// Immutable schedules, prepared once per complete fit. dist2 contains only
// distinct normalized distance matrices; subject_group indexes those matrices.
struct Geometry {
  std::vector<arma::vec> times;
  std::vector<arma::mat> dist2;
  std::vector<arma::uvec> positive;
  std::vector<arma::vec> log_dist2;
  std::vector<arma::uword> rank;
  std::vector<std::size_t> subject_group;
  arma::vec counts;
  double gap = 1;
  double median_gap = 1;
  double max_gap = 1;
  std::vector<double> scales;
  bool repeated = false;
  bool has_gaps = false;
};

Geometry prepare_geometry(const std::vector<arma::vec>& times);
void rbf_kernel(const Geometry& geometry, std::size_t group, double log_length,
                arma::mat& correlation, arma::mat* derivative = nullptr);

struct PriorResult {
  arma::vec mean;
  arma::vec second;
  Rcpp::RObject par = R_NilValue;
  double loglik = 0;
  double neg_kl = 0;
  Rcpp::List extra = Rcpp::List::create();
};

PriorResult gaussian_update(const arma::vec& x, double s);
PriorResult fit_aligned(const arma::mat& X, const arma::vec& s,
                        const Geometry& geometry, const std::string& prior,
                        SEXP par_init = R_NilValue);
PriorResult fit_irregular(const std::vector<arma::vec>& X, double s,
                          const Geometry& geometry, const std::string& prior,
                          SEXP par_init = R_NilValue);

// Aligned PriorResult means use vectorise(subjects x timepoints), as in R.
// Irregular means concatenate subjects' visits. Both store E[Z^2], not variance.
Rcpp::List prior_output(const PriorResult& fit, const std::string& prior);

struct WorkingData {
  arma::mat X; // Variables x visits, with each subject's visits contiguous.
  arma::uvec offsets;
  arma::uvec noise_group;
  arma::vec group_size;
  arma::uvec missing;
  Geometry geometry;
  bool aligned;
  arma::uword subjects;
  arma::uword visits;
};

struct Factors {
  arma::mat U, U2, V, V2;
  arma::vec u_loglik, v_loglik, u_kl, v_kl;
  std::vector<Rcpp::RObject> u_par, v_par, v_diagnostics;
  std::vector<int> v_converged;
  arma::uword rank = 0;
  arma::uword u_count = 0, v_count = 0;
  bool diagnostics = false;
};

struct ResidualCache {
  arma::mat residual;
  arma::vec squared_error;
  arma::vec uncertainty;
  arma::vec rss;
};

Factors empty_factors(arma::uword variables, arma::uword visits, arma::uword capacity);
Factors initialize_factors(const WorkingData& data, arma::uword rank);
ResidualCache initialize_residual(const WorkingData& data, const Factors& factors);
void replace_loading(const WorkingData& data, Factors& factors, ResidualCache& cache,
                     arma::uword k, const PriorResult& update);
void replace_score(const WorkingData& data, Factors& factors, ResidualCache& cache,
                   arma::uword k, const PriorResult& update);
double evidence_lower_bound(const WorkingData& data, const Factors& factors,
                            const ResidualCache& cache, const arma::vec& tau);

} // namespace evblm
#endif
