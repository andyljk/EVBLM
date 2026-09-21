#include "helper.h"
#include <algorithm>

namespace evblm {

Evaluation irregular_ec_objective(const arma::vec& par, const arma::vec& counts,
                                 const arma::vec& parallel, const arma::vec& contrast,
                                 double noise) {
  const double v = 1 + par[0] / noise;
  const arma::vec w = 1 + (par[0] + counts * par[1]) / noise;
  const double denominator = 2 * arma::accu(counts);
  const arma::vec common = 1 / w - parallel / arma::square(w);
  Evaluation result;
  result.value = arma::accu((counts - 1) * std::log(v) + arma::log(w) +
    contrast / v + parallel / w) / denominator;
  result.gradient = arma::vec({arma::accu((counts - 1) / v - contrast / (v * v) + common),
    arma::dot(counts, common)}) / (denominator * noise);
  return result;
}

// The schedule geometry is fixed across factor updates. Within one update,
// subjects with the same geometry contribute only their summed crossproducts.
class IrregularRbfObjective {
 public:
  const Geometry& geometry;
  std::vector<arma::mat> crossproducts;
  std::vector<double> group_counts;
  double noise;
  double total;

  IrregularRbfObjective(const Geometry& geometry_, const std::vector<arma::vec>& y,
                       double noise_) : geometry(geometry_), noise(noise_),
                       total(arma::accu(geometry.counts)) {
    const std::size_t groups = geometry.dist2.size();
    crossproducts.resize(groups);
    group_counts.assign(groups, 0);
    for (std::size_t g = 0; g < groups; ++g) {
      const arma::mat& d = geometry.dist2[g];
      crossproducts[g].zeros(d.n_rows, d.n_cols);
    }
    for (std::size_t i = 0; i < y.size(); ++i) {
      const std::size_t g = geometry.subject_group[i];
      crossproducts[g] += y[i] * y[i].t();
      group_counts[g] += 1;
    }
  }

  Evaluation operator()(const arma::vec& par) const {
    Evaluation result;
    result.value = 0;
    result.gradient.zeros(2);
    for (std::size_t g = 0; g < crossproducts.size(); ++g) {
      const arma::mat& dist2 = geometry.dist2[g];
      const arma::uword m = dist2.n_rows;
      arma::mat correlation, derivative;
      rbf_kernel(geometry, g, par[1], correlation, &derivative);
      arma::vec eigenvalues;
      arma::mat vectors;
      arma::eig_sym(eigenvalues, vectors, correlation);
      eigenvalues = arma::reverse(eigenvalues);
      vectors = arma::fliplr(vectors);
      arma::vec h = arma::clamp(eigenvalues, 0, arma::datum::inf) / noise;
      arma::uword rank = geometry.rank[g];
      if (arma::all(arma::vectorise(correlation == 1))) rank = 1;
      if (rank < m) h.tail(m - rank).zeros();
      const arma::vec lambda = par[0] * h;
      const arma::vec inverse_values = 1 / (1 + lambda);
      const arma::mat rotated = vectors.t() * crossproducts[g] * vectors;
      const arma::vec energy = rotated.diag();
      result.value += group_counts[g] * arma::accu(arma::log1p(lambda)) +
        arma::dot(energy, inverse_values);
      result.gradient[0] += arma::dot(h,
        group_counts[g] * inverse_values - energy % arma::square(inverse_values));
      const arma::mat inverse = (vectors.each_row() % inverse_values.t()) * vectors.t();
      result.gradient[1] += par[0] / noise * arma::accu((group_counts[g] * inverse -
        inverse * crossproducts[g] * inverse) % derivative);
    }
    result.value /= 2 * total;
    result.gradient /= 2 * total;
    return result;
  }
};

PriorResult fit_irregular(const std::vector<arma::vec>& observations, double s,
                          const Geometry& geometry, const std::string& prior,
                          SEXP par_init) {
  const arma::vec& counts = geometry.counts;
  const std::size_t n = observations.size();
  PriorResult output;
  Rcpp::List& results = output.extra;
  output.mean.zeros(arma::accu(counts));
  output.second.zeros(arma::accu(counts));
  if (std::isinf(s)) {
    output.par = Rcpp::NumericVector::create(Rcpp::Named("eta") = 0,
      Rcpp::Named(prior == "EC" ? "rho" : "alpha") = NA_REAL);
    results["solver"] = "uninformative block";
    results["converged"] = true;
    return output;
  }
  const std::vector<arma::vec>& x = observations;
  std::vector<arma::vec> y(n);
  arma::vec parallel(n), contrast(n);
  double sum_squares = 0;
  for (std::size_t i = 0; i < n; ++i) {
    sum_squares += arma::dot(x[i], x[i]);
    y[i] = x[i] / s;
    parallel[i] = std::pow(arma::accu(y[i]), 2) / counts[i];
    contrast[i] = arma::accu(arma::square(y[i] - arma::mean(y[i])));
  }
  const double mean_square = sum_squares / arma::accu(counts);
  const double reference = mean_square + s * s;
  const double noise = s * s / reference;
  const double pgtol = prior == "EC" ? 1e-6 : 1e-7;
  double independent = std::max(mean_square - s * s, 0.0) / reference;
  auto ec = [&](const arma::vec& par) {
    return irregular_ec_objective(par, counts, parallel, contrast, noise);
  };
  const bool equal_counts = arma::all(counts == counts[0]);
  double constant = std::max(arma::mean(parallel / counts) - arma::mean(1 / counts), 0.0) * noise;
  int constant_code = 0;
  if (equal_counts) {
    constant = std::max(arma::mean(parallel) - 1, 0.0) * noise / counts[0];
  } else {
    auto objective = [&](const arma::vec& b) {
      Evaluation value = ec(arma::vec({0, b[0]}));
      value.gradient = arma::vec({value.gradient[1]});
      return value;
    };
    OptimResult opt = optimize_lbfgsb(arma::vec({constant}),
      arma::vec({0}), arma::vec({arma::datum::inf}), objective, 500, 100, pgtol);
    constant = opt.par[0];
    constant_code = opt.code;
  }
  std::vector<std::string> diagnostic_names = {"constant"};
  std::vector<int> diagnostic_codes = {constant_code};
  std::vector<double> diagnostic_values = {ec(arma::vec({0, constant})).value};
  arma::vec best(2, arma::fill::zeros);
  std::string selected;
  double variance = 0;
  double alpha = NA_REAL;
  bool has_previous = par_init != R_NilValue;
  arma::vec previous;
  if (has_previous) previous = Rcpp::as<arma::vec>(par_init);
  if (prior == "EC") {
    if (equal_counts) {
      const double m = counts[0];
      if (m == 1) best = arma::vec({independent, 0});
      else {
        const double vc = arma::accu(contrast) / (n * (m - 1));
        const double vp = arma::mean(parallel);
        const double a = std::max(vc - 1, 0.0) * noise;
        best = vp >= vc ? arma::vec({a, std::max((vp - 1) * noise - a, 0.0) / m}) :
          arma::vec({independent, 0});
      }
      selected = "closed form";
      results["solver"] = "closed form";
      diagnostic_names = {selected};
      diagnostic_codes = {0};
      diagnostic_values = {ec(best).value};
    } else {
      std::vector<arma::vec> candidates = {arma::vec({0, 0}), arma::vec({independent, 0}),
        arma::vec({0, constant})};
      std::vector<std::string> names = {"zero", "independent", "constant"};
      const double a = std::max(arma::accu(contrast) / arma::accu(counts - 1) - 1, 0.0) * noise;
      std::vector<arma::vec> starts = {
        arma::vec({a, std::max(constant - a / arma::mean(counts), 0.0)}),
        arma::vec({independent / 2, independent / 2}), arma::vec({0, constant})};
      if (has_previous && previous[0] > 0) {
        const double rho = std::isnan(previous[1]) ? 0 : previous[1];
        arma::vec previous_scaled = previous[0] * previous[0] / reference * arma::vec({1 - rho, rho});
        candidates.push_back(previous_scaled);
        names.push_back("previous");
        starts.insert(starts.begin(), previous_scaled);
      }
      for (std::size_t j = 0; j < starts.size(); ++j) {
        OptimResult opt = optimize_lbfgsb(starts[j], arma::vec({0, 0}),
          arma::vec({arma::datum::inf, arma::datum::inf}), ec, 500, 100, pgtol);
        std::string name = "start" + std::to_string(j + 1);
        candidates.push_back(opt.par);
        names.push_back(name);
        diagnostic_names.push_back(name);
        diagnostic_codes.push_back(opt.code);
        diagnostic_values.push_back(opt.value);
      }
      double minimum = arma::datum::inf;
      for (std::size_t j = 0; j < candidates.size(); ++j) {
        double value = ec(candidates[j]).value;
        if (value < minimum) {
          minimum = value;
          best = candidates[j];
          selected = names[j];
        }
      }
      results["solver"] = "L-BFGS-B with analytical gradient";
    }
    variance = reference * arma::accu(best);
    const double rho = variance == 0 || arma::all(counts == 1) ? NA_REAL : best[1] / arma::accu(best);
    output.par = Rcpp::NumericVector::create(Rcpp::Named("eta") = std::sqrt(variance),
      Rcpp::Named("rho") = rho);
    arma::vec score = ec(best).gradient;
    for (arma::uword j = 0; j < 2; ++j) if (best[j] == 0) score[j] = std::min(score[j], 0.0);
    results["projected_gradient"] = arma::max(arma::abs(score));
  } else {
    if (!geometry.has_gaps) {
      variance = reference * constant;
      selected = "constant";
      results["solver"] = "constant process; length unidentified";
    } else {
      IrregularRbfObjective rbf(geometry, y, noise);
      int independent_code = 0;
      if (geometry.repeated) {
        auto objective = [&](const arma::vec& a) {
          Evaluation value = rbf(arma::vec({a[0], -arma::datum::inf}));
          value.gradient = arma::vec({value.gradient[0]});
          return value;
        };
        OptimResult opt = optimize_lbfgsb(arma::vec({independent}),
          arma::vec({0}), arma::vec({arma::datum::inf}), objective, 500, 100, pgtol);
        independent = opt.par[0];
        independent_code = opt.code;
      }
      std::vector<arma::vec> candidates = {arma::vec({0, 0}),
        arma::vec({independent, -arma::datum::inf}), arma::vec({constant, arma::datum::inf})};
      std::vector<std::string> names = {"zero", "independent", "constant"};
      diagnostic_names.push_back("independent");
      diagnostic_codes.push_back(independent_code);
      diagnostic_values.push_back(rbf(candidates[1]).value);
      std::vector<arma::vec> starts;
      for (double scale : geometry.scales) {
        starts.push_back(arma::vec({mean_square / reference, std::log(scale / geometry.gap)}));
      }
      if (has_previous && previous[0] > 0 && !std::isnan(previous[1])) {
        arma::vec previous_scaled = {previous[0] * previous[0] / reference,
          std::log(previous[1] / geometry.gap)};
        candidates.push_back(previous_scaled);
        names.push_back("previous");
        if (std::isfinite(previous_scaled[1])) starts.insert(starts.begin(), previous_scaled);
      }
      for (std::size_t j = 0; j < starts.size(); ++j) {
        OptimResult opt = optimize_lbfgsb(starts[j], arma::vec({0, -arma::datum::inf}),
          arma::vec({arma::datum::inf, arma::datum::inf}),
          [&](const arma::vec& par) { return rbf(par); }, 500, 100, pgtol);
        std::string name = "start" + std::to_string(j + 1);
        candidates.push_back(opt.par);
        names.push_back(name);
        diagnostic_names.push_back(name);
        diagnostic_codes.push_back(opt.code);
        diagnostic_values.push_back(opt.value);
      }
      double minimum = arma::datum::inf;
      for (std::size_t j = 0; j < candidates.size(); ++j) {
        double value = rbf(candidates[j]).value;
        if (value < minimum) {
          minimum = value;
          best = candidates[j];
          selected = names[j];
        }
      }
      variance = reference * best[0];
      alpha = variance == 0 ? NA_REAL : geometry.gap * std::exp(best[1]);
      arma::vec score = rbf(best).gradient;
      if (best[0] == 0) score[0] = std::min(score[0], 0.0);
      results["projected_gradient"] = std::isfinite(best[1]) ? arma::max(arma::abs(score)) : std::abs(score[0]);
      results["solver"] = "L-BFGS-B with analytical gradient";
    }
    output.par = Rcpp::NumericVector::create(Rcpp::Named("eta") = std::sqrt(variance),
      Rcpp::Named("alpha") = alpha);
  }
  results["optimization"] = Rcpp::DataFrame::create(Rcpp::Named("candidate") = diagnostic_names,
    Rcpp::Named("code") = diagnostic_codes, Rcpp::Named("objective") = diagnostic_values);
  results["selected_start"] = selected;
  int converged = NA_LOGICAL;
  for (std::size_t j = 0; j < diagnostic_names.size(); ++j) {
    if (selected == diagnostic_names[j]) {
      converged = diagnostic_codes[j] == 0;
      break;
    }
  }
  if ((selected == "zero" || selected == "independent") && converged == NA_LOGICAL) converged = true;
  if (selected == "previous") converged = NA_LOGICAL;
  results["converged"] = Rcpp::LogicalVector::create(converged);
  double loglik = 0;
  double neg_kl = 0;
  arma::uword offset = 0;
  const double log_noise = std::log(2 * arma::datum::pi * s * s);
  if (prior == "EC") {
    for (std::size_t i = 0; i < n; ++i) {
      const double m = counts[i];
      const double lambda_contrast = best[0] / noise;
      const double lambda_parallel = (best[0] + m * best[1]) / noise;
      const double shrink_contrast = lambda_contrast / (1 + lambda_contrast);
      const double shrink_parallel = lambda_parallel / (1 + lambda_parallel);
      arma::vec mu = shrink_contrast * x[i] + (shrink_parallel - shrink_contrast) * arma::mean(x[i]);
      const double posterior_variance = s * s * ((m - 1) * shrink_contrast + shrink_parallel) / m;
      output.mean.subvec(offset, offset + mu.n_elem - 1) = mu;
      output.second.subvec(offset, offset + mu.n_elem - 1) = arma::square(mu) + posterior_variance;
      offset += mu.n_elem;
      loglik -= 0.5 * (m * log_noise + (m - 1) * std::log1p(lambda_contrast) +
        std::log1p(lambda_parallel) + contrast[i] / (1 + lambda_contrast) + parallel[i] / (1 + lambda_parallel));
      neg_kl -= 0.5 * ((m - 1) * (std::log1p(lambda_contrast) - shrink_contrast) +
        std::log1p(lambda_parallel) - shrink_parallel + contrast[i] * shrink_contrast / (1 + lambda_contrast) +
        parallel[i] * shrink_parallel / (1 + lambda_parallel));
    }
  } else {
    std::vector<arma::mat> vectors(geometry.dist2.size());
    std::vector<arma::vec> lambda(geometry.dist2.size()), shrink(geometry.dist2.size());
    std::vector<arma::vec> posterior_variance(geometry.dist2.size());
    for (std::size_t g = 0; g < geometry.dist2.size(); ++g) {
      const arma::mat& dist2 = geometry.dist2[g];
      const arma::uword m = dist2.n_rows;
      arma::mat covariance(m, m, arma::fill::value(variance));
      if (variance > 0 && !std::isnan(alpha)) {
        rbf_kernel(geometry, g, best[1], covariance);
        covariance *= variance;
      }
      arma::vec eigenvalues;
      arma::eig_sym(eigenvalues, vectors[g], covariance);
      eigenvalues = arma::reverse(eigenvalues);
      vectors[g] = arma::fliplr(vectors[g]);
      lambda[g] = arma::clamp(eigenvalues, 0, arma::datum::inf) / (s * s);
      arma::uword rank = geometry.rank[g];
      if (arma::all(arma::vectorise(covariance == covariance[0]))) rank = 1;
      if (rank < m) lambda[g].tail(m - rank).zeros();
      shrink[g] = lambda[g] / (1 + lambda[g]);
      posterior_variance[g] = s * s * arma::square(vectors[g]) * shrink[g];
    }
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t g = geometry.subject_group[i];
      const arma::vec z = vectors[g].t() * y[i];
      const arma::vec mu = s * vectors[g] * (shrink[g] % z);
      output.mean.subvec(offset, offset + mu.n_elem - 1) = mu;
      output.second.subvec(offset, offset + mu.n_elem - 1) = arma::square(mu) + posterior_variance[g];
      offset += mu.n_elem;
      loglik -= 0.5 * (counts[i] * log_noise + arma::accu(arma::log1p(lambda[g]) + arma::square(z) / (1 + lambda[g])));
      neg_kl -= 0.5 * arma::accu(arma::log1p(lambda[g]) - shrink[g] + arma::square(z) % shrink[g] / (1 + lambda[g]));
    }
  }
  output.loglik = loglik;
  output.neg_kl = neg_kl;
  return output;
}

} // namespace evblm
