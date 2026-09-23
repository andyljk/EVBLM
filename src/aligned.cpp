#include "helper.h"

namespace evblm {

PriorResult fit_aligned(const arma::mat& X, const arma::vec& s,
                        const Geometry& geometry, const std::string& prior,
                        SEXP par_init) {
  const arma::uword p = X.n_rows;
  const arma::uword d = X.n_cols;
  PriorResult result;
  result.mean.zeros(p * d);
  result.second.zeros(p * d);
  bool uninformative = true;
  for (double standard_error : s) {
    if (!std::isinf(standard_error)) uninformative = false;
  }
  if (uninformative) {
    if (prior == "Free") result.par = Rcpp::wrap(arma::mat(d, d, arma::fill::zeros));
    else if (prior == "EC") {
      result.par = Rcpp::NumericVector::create(Rcpp::Named("eta") = 0,
                                              Rcpp::Named("rho") = 0);
    } else {
      result.par = Rcpp::NumericVector::create(Rcpp::Named("eta") = 0,
                                              Rcpp::Named("alpha") = NA_REAL);
    }
    return result;
  }

  const arma::vec s2 = arma::square(s);
  const arma::mat sample_cov = X.t() * X / static_cast<double>(p);
  arma::mat V(d, d, arma::fill::zeros);

  if (prior == "EC") {
    double parallel = 0;
    double perpendicular = 0;
    if (d == 1 || arma::all(s == s[0])) {
      parallel = std::max(arma::accu(sample_cov) / d - s2[0], 0.0);
      if (d > 1) {
        perpendicular = std::max((arma::trace(sample_cov) -
          arma::accu(sample_cov) / d) / (d - 1) - s2[0], 0.0);
      }
      result.extra = Rcpp::List::create(Rcpp::Named("solver") = "closed form",
        Rcpp::Named("n_iter") = 0, Rcpp::Named("converged") = true);
    } else {
      // V = k_perp I + b 11'. b may be negative for aligned EC priors.
      // V + S remains positive definite at both semidefinite prior boundaries.
      const arma::vec covariance_diagonal = sample_cov.diag();
      const arma::vec noise_precision = 1 / s2;
      const double h = arma::accu(noise_precision) / d;
      const double z = arma::dot(noise_precision, sample_cov * noise_precision) / d;
      double best_loglik = -0.5 * (arma::accu(arma::log(s2)) +
        arma::dot(covariance_diagonal, noise_precision));
      const double boundary_parallel = std::max((z - h) / (h * h), 0.0);

      // Matrix determinant lemma and Sherman-Morrison eliminate factorizations.
      auto ec_loglik = [&](double k_parallel, double k_perp) {
        const arma::vec diagonal = s2 + k_perp;
        const arma::vec q = 1 / diagonal;
        const double b = (k_parallel - k_perp) / d;
        const double determinant_ratio = arma::mean((s2 + k_parallel) % q);
        const double coefficient = b / determinant_ratio;
        return -0.5 * (arma::accu(arma::log(diagonal)) +
          std::log(determinant_ratio) + arma::dot(covariance_diagonal, q) -
          coefficient * arma::dot(q, sample_cov * q));
      };
      const double boundary_loglik = ec_loglik(boundary_parallel, 0);
      if (boundary_loglik > best_loglik) {
        parallel = boundary_parallel;
        best_loglik = boundary_loglik;
      }
      std::vector<arma::vec> starts;
      if (par_init != R_NilValue) {
        const Rcpp::NumericVector initial(par_init);
        const double variance = initial[0] * initial[0];
        starts.push_back(arma::vec({variance * (1 + (d - 1) * initial[1]),
                                   variance * (1 - initial[1])}));
      }
      const double mean_variance = arma::trace(sample_cov) / d;
      starts.push_back(arma::vec({mean_variance, mean_variance}));
      starts.push_back(arma::vec({0, mean_variance}));
      int best_iteration = 0;
      bool best_converged = true;
      for (const arma::vec& start : starts) {
        double current_parallel = start[0];
        double current_perpendicular = start[1];
        double loglik = ec_loglik(current_parallel, current_perpendicular);
        bool converged = false;
        int iteration = 0;
        for (iteration = 1; iteration <= 1000; ++iteration) {
          // B = V(V+S)^-1 = diag(a) + coefficient * u q'.
          // Only tr(E[zz']) and e'E[zz']e enter the EC M-step.
          const arma::vec q = 1 / (s2 + current_perpendicular);
          const arma::vec u = s2 % q;
          const arma::vec a = current_perpendicular * q;
          const double b = (current_parallel - current_perpendicular) / d;
          const double determinant_ratio = arma::mean((s2 + current_parallel) % q);
          const double coefficient = b / determinant_ratio;
          const arma::vec covariance_q = sample_cov * q;
          const arma::vec mean_direction =
            (a + coefficient * q * arma::accu(u)) / std::sqrt(static_cast<double>(d));
          const double diagonal_variance = arma::dot(s2, a);
          const double trace_variance = diagonal_variance + coefficient * arma::dot(u, u);
          const double parallel_variance = (diagonal_variance +
            coefficient * std::pow(arma::accu(u), 2)) / d;
          const double trace_second = trace_variance +
            arma::dot(arma::square(a), covariance_diagonal) +
            2 * coefficient * arma::dot(a % u, covariance_q) +
            coefficient * coefficient * arma::dot(u, u) * arma::dot(q, covariance_q);
          const double parallel_second = parallel_variance +
            arma::dot(mean_direction, sample_cov * mean_direction);
          current_parallel = std::max(parallel_second, 0.0);
          current_perpendicular = std::max((trace_second - parallel_second) / (d - 1), 0.0);
          const double new_loglik = ec_loglik(current_parallel, current_perpendicular);
          converged = std::abs(new_loglik - loglik) <= 1e-10 * (1 + std::abs(loglik));
          loglik = new_loglik;
          if (converged) break;
          if (iteration % 50 == 0) Rcpp::checkUserInterrupt();
        }
        if (loglik > best_loglik) {
          parallel = current_parallel;
          perpendicular = current_perpendicular;
          best_loglik = loglik;
          best_iteration = std::min(iteration, 1000);
          best_converged = converged;
        }
      }
      result.extra = Rcpp::List::create(Rcpp::Named("solver") = "analytical EM",
        Rcpp::Named("n_iter") = best_iteration,
        Rcpp::Named("converged") = best_converged);
    }
    V.fill((parallel - perpendicular) / d);
    V.diag() += perpendicular;
    const double variance = (parallel + (d - 1) * perpendicular) / d;
    const double rho = variance == 0 || d == 1 ? 0 :
      (parallel - perpendicular) / (d * variance);
    result.par = Rcpp::NumericVector::create(Rcpp::Named("eta") = std::sqrt(variance),
                                            Rcpp::Named("rho") = rho);
  } else if (prior == "Free") {
    arma::mat whitened_covariance = sample_cov;
    whitened_covariance.each_col() /= s;
    whitened_covariance.each_row() /= s.t();
    arma::vec eigenvalues;
    arma::mat eigenvectors;
    if (!arma::eig_sym(eigenvalues, eigenvectors, whitened_covariance))
      Rcpp::stop("Aligned Free covariance eigendecomposition failed.");
    eigenvalues = arma::clamp(eigenvalues - 1, 0, arma::datum::inf);
    V = (eigenvectors.each_row() % eigenvalues.t()) * eigenvectors.t();
    V.each_col() %= s;
    V.each_row() %= s.t();
    result.par = Rcpp::wrap(V);
    result.extra = Rcpp::List::create(Rcpp::Named("solver") = "closed form",
      Rcpp::Named("n_iter") = 0, Rcpp::Named("converged") = true);
  } else if (prior == "RBF") {
    // The likelihood sees X only through its standardized second moment.
    arma::mat standardized_covariance = sample_cov;
    standardized_covariance.each_col() /= s;
    standardized_covariance.each_row() /= s.t();
    const double reference = (arma::trace(sample_cov) + arma::accu(s2)) / d;
    const arma::vec scaled_s = s / std::sqrt(reference);
    const arma::vec precision = 1 / s2;
    const double h = arma::accu(precision);
    const double constant_variance = std::max(
      (arma::dot(precision, sample_cov * precision) - h) / (h * h), 0.0);
    result.extra = Rcpp::List::create(
      Rcpp::Named("solver") = "L-BFGS-B with analytical gradient");
    if (!geometry.has_gaps) {
      V.fill(constant_variance);
      result.par = Rcpp::NumericVector::create(
        Rcpp::Named("eta") = std::sqrt(constant_variance),
        Rcpp::Named("alpha") = R_PosInf);
      result.extra["converged"] = true;
    } else {
      Objective objective = [&](const arma::vec& par) {
        arma::mat correlation, derivative;
        rbf_kernel(geometry, 0, par[1], correlation, &derivative);
        arma::mat H = correlation;
        H.each_col() /= scaled_s;
        H.each_row() /= scaled_s.t();
        arma::vec eigenvalues;
        arma::mat eigenvectors;
        if (!arma::eig_sym(eigenvalues, eigenvectors, H))
          Rcpp::stop("Aligned RBF objective eigendecomposition failed.");
        eigenvalues = arma::clamp(eigenvalues, 0, arma::datum::inf);
        if (arma::all(arma::vectorise(correlation) == 1) && d > 1) {
          eigenvalues.head(d - 1).zeros();
        }
        const arma::vec lambda = par[0] * eigenvalues;
        const arma::vec inverse_values = 1 / (1 + lambda);
        const arma::vec energy = arma::sum(eigenvectors %
          (standardized_covariance * eigenvectors), 0).t();
        const arma::mat inverse =
          (eigenvectors.each_row() % inverse_values.t()) * eigenvectors.t();
        derivative.each_col() /= scaled_s;
        derivative.each_row() /= scaled_s.t();
        Evaluation evaluation;
        evaluation.value = arma::accu(arma::log1p(lambda) + energy % inverse_values) / (2 * d);
        evaluation.gradient.set_size(2);
        evaluation.gradient[0] = arma::dot(eigenvalues,
          inverse_values - energy % arma::square(inverse_values)) / (2 * d);
        evaluation.gradient[1] = par[0] * arma::accu((inverse -
          inverse * standardized_covariance * inverse) % derivative) / (2 * d);
        return evaluation;
      };
      const double amplitude = arma::trace(sample_cov) / (d * reference);
      Objective independent_objective = [&](const arma::vec& amplitude_par) {
        Evaluation evaluation = objective(arma::vec({amplitude_par[0], -arma::datum::inf}));
        evaluation.gradient = evaluation.gradient.head(1);
        return evaluation;
      };
      const OptimResult independent = optimize_lbfgsb(arma::vec({amplitude}),
        arma::vec({0}), arma::vec({arma::datum::inf}), independent_objective);
      std::vector<arma::vec> candidates = {
        arma::vec({0, 0}), arma::vec({independent.par[0], -arma::datum::inf}),
        arma::vec({constant_variance / reference, arma::datum::inf})};
      std::vector<std::string> candidate_names = {"zero", "independent", "constant"};
      std::vector<arma::vec> starts;
      const std::vector<double> scales = {geometry.gap, geometry.median_gap, geometry.max_gap};
      std::vector<double> used_scales;
      for (double scale : scales) {
        if (std::find(used_scales.begin(), used_scales.end(), scale) == used_scales.end()) {
          starts.push_back(arma::vec({amplitude, std::log(scale / geometry.gap)}));
          used_scales.push_back(scale);
        }
      }
      if (par_init != R_NilValue) {
        const Rcpp::NumericVector initial(par_init);
        if (initial[0] > 0 && !Rcpp::NumericVector::is_na(initial[1])) {
          const arma::vec previous = {initial[0] * initial[0] / reference,
            std::log(initial[1] / geometry.gap)};
          candidates.push_back(previous);
          candidate_names.push_back("previous");
          if (std::isfinite(previous[1])) starts.insert(starts.begin(), previous);
        }
      }
      std::vector<std::string> diagnostic_names = {"independent"};
      std::vector<int> diagnostic_codes = {independent.code};
      std::vector<double> diagnostic_values = {independent.value};
      std::vector<double> diagnostic_gradients = {
        std::abs(independent_objective(independent.par).gradient[0])};
      std::vector<int> diagnostic_evaluations = {independent.fncount + independent.grcount};
      for (std::size_t i = 0; i < starts.size(); ++i) {
        const OptimResult optimum = optimize_lbfgsb(starts[i],
          arma::vec({0, -arma::datum::inf}),
          arma::vec({arma::datum::inf, arma::datum::inf}), objective);
        arma::vec score = objective(optimum.par).gradient;
        if (optimum.par[0] == 0) score[0] = std::min(score[0], 0.0);
        const std::string name = "start" + std::to_string(i + 1);
        diagnostic_names.push_back(name);
        diagnostic_codes.push_back(optimum.code);
        diagnostic_values.push_back(optimum.value);
        diagnostic_gradients.push_back(arma::abs(score).max());
        diagnostic_evaluations.push_back(optimum.fncount + optimum.grcount);
        candidates.push_back(optimum.par);
        candidate_names.push_back(name);
      }
      std::size_t selected = 0;
      double best_value = objective(candidates[0]).value;
      for (std::size_t i = 1; i < candidates.size(); ++i) {
        const double value = objective(candidates[i]).value;
        if (value < best_value) {
          best_value = value;
          selected = i;
        }
      }
      const arma::vec& best = candidates[selected];
      const double variance = reference * best[0];
      const double alpha = variance == 0 ? NA_REAL : geometry.gap * std::exp(best[1]);
      V.fill(variance);
      if (variance > 0) {
        rbf_kernel(geometry, 0, best[1], V);
        V *= variance;
      }
      result.par = Rcpp::NumericVector::create(Rcpp::Named("eta") = std::sqrt(variance),
                                              Rcpp::Named("alpha") = alpha);
      result.extra["optimization"] = Rcpp::DataFrame::create(
        Rcpp::Named("candidate") = diagnostic_names, Rcpp::Named("code") = diagnostic_codes,
        Rcpp::Named("objective") = diagnostic_values,
        Rcpp::Named("gradient") = diagnostic_gradients,
        Rcpp::Named("evaluations") = diagnostic_evaluations);
      const std::string& selected_name = candidate_names[selected];
      result.extra["selected_start"] = selected_name;
      int converged = TRUE;
      if (selected_name == "previous") converged = NA_LOGICAL;
      else if (selected_name != "zero" && selected_name != "constant") {
        const auto match = std::find(diagnostic_names.begin(), diagnostic_names.end(), selected_name);
        converged = diagnostic_codes[match - diagnostic_names.begin()] == 0;
      }
      result.extra["converged"] = Rcpp::LogicalVector::create(converged);
    }
  }

  // Whitened conditioning preserves singular priors and avoids V - V A^-1 V.
  arma::mat standardized_prior = V;
  standardized_prior.each_col() /= s;
  standardized_prior.each_row() /= s.t();
  arma::vec lambda;
  arma::mat eigenvectors;
  if (!arma::eig_sym(lambda, eigenvectors, standardized_prior))
    Rcpp::stop("Aligned posterior covariance eigendecomposition failed.");
  lambda = arma::clamp(lambda, 0, arma::datum::inf);
  if (arma::all(arma::vectorise(V) == V[0]) && d > 1) lambda.head(d - 1).zeros();
  const arma::vec shrink = lambda / (1 + lambda);
  arma::mat standardized_X = X;
  standardized_X.each_row() /= s.t();
  const arma::mat Z = standardized_X * eigenvectors;
  arma::mat posterior_mean = (Z.each_row() % shrink.t()) * eigenvectors.t();
  posterior_mean.each_row() %= s.t();
  const arma::vec posterior_variance = s2 % (arma::square(eigenvectors) * shrink);
  arma::mat posterior_second = arma::square(posterior_mean);
  posterior_second.each_row() += posterior_variance.t();
  result.mean = arma::vectorise(posterior_mean);
  result.second = arma::vectorise(posterior_second);
  const arma::vec energy = arma::sum(arma::square(Z), 0).t();
  result.loglik = -0.5 * (p * (arma::accu(arma::log(2 * arma::datum::pi * s2)) +
    arma::accu(arma::log1p(lambda))) + arma::accu(energy / (1 + lambda)));
  result.neg_kl = -0.5 * arma::accu(p * (arma::log1p(lambda) - shrink) +
    energy % shrink / (1 + lambda));
  return result;
}

} // namespace evblm
