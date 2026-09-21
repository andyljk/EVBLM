#include "helper.h"

// [[Rcpp::export]]
Rcpp::List mv_ebnm_cpp(const arma::mat& X, const arma::vec& s,
                      const arma::vec& D, const std::string& prior,
                      SEXP par_init = R_NilValue) {
  evblm::Geometry geometry = evblm::prepare_geometry({D});
  evblm::PriorResult fit = evblm::fit_aligned(X, s, geometry, prior, par_init);
  Rcpp::List output = evblm::prior_output(fit, prior);
  output["mean"] = fit.mean;
  output["second"] = fit.second;
  return output;
}

// [[Rcpp::export]]
Rcpp::List irr_ebnm_cpp(const Rcpp::List& X, double s, const Rcpp::List& D,
                       const std::string& prior, SEXP par_init = R_NilValue) {
  std::vector<arma::vec> observations(X.size()), times(D.size());
  for (int i = 0; i < X.size(); ++i) {
    observations[i] = Rcpp::as<arma::vec>(X[i]);
    times[i] = Rcpp::as<arma::vec>(D[i]);
  }
  evblm::Geometry geometry = evblm::prepare_geometry(times);
  evblm::PriorResult fit = evblm::fit_irregular(observations, s, geometry, prior, par_init);
  Rcpp::List output = evblm::prior_output(fit, prior);
  output["mean"] = fit.mean;
  output["second"] = fit.second;
  return output;
}
