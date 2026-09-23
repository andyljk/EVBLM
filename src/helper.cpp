#include "helper.h"
#include <R_ext/Applic.h>
#include <algorithm>
#include <cmath>

namespace evblm {
namespace {

// Apply the smaller Gram operator using two matrix-vector products. Neither
// X'X nor XX' is formed; NEWARP supplies restarted Lanczos and a local RNG.
struct SvdGramOperator {
  const arma::mat& X;
  const bool right;
  const arma::uword n_rows, n_cols;
  const double scale;
  mutable arma::vec work;

  SvdGramOperator(const arma::mat& matrix) : X(matrix), right(X.n_cols <= X.n_rows),
    n_rows(std::min(X.n_rows, X.n_cols)), n_cols(n_rows), scale(arma::abs(X).max()) {}

  void perform_op(double* input, double* output) const {
    Rcpp::checkUserInterrupt();
    const arma::vec x(input, n_rows, false, true);
    arma::vec y(output, n_rows, false, true);
    if (right) {
      work = (X * x) / scale;
      y = (X.t() * work) / scale;
    } else {
      work = (X.t() * x) / scale;
      y = (X * work) / scale;
    }
  }
};

// optim requests fn and gr separately. Retain their shared evaluation at x.
struct OptimContext {
  const Objective& objective;
  arma::vec x;
  Evaluation evaluation;
  bool available = false;

  const Evaluation& evaluate(int n, double* values) {
    arma::vec next(values, n, false, true);
    if (!available || x.n_elem != next.n_elem || !arma::all(x == next)) {
      x = next;
      evaluation = objective(x);
      available = true;
    }
    return evaluation;
  }
};

double objective_callback(int n, double* x, void* pointer) {
  return static_cast<OptimContext*>(pointer)->evaluate(n, x).value;
}

void gradient_callback(int n, double* x, double* gradient, void* pointer) {
  const auto& evaluation = static_cast<OptimContext*>(pointer)->evaluate(n, x);
  std::copy(evaluation.gradient.begin(), evaluation.gradient.end(), gradient);
}
} // namespace

OptimResult optimize_lbfgsb(const arma::vec& initial, const arma::vec& lower,
                           const arma::vec& upper, const Objective& objective,
                           int maxit, double factr, double pgtol) {
  OptimResult result;
  result.par = initial;
  arma::vec lo = lower, hi = upper;
  std::vector<int> bounds(initial.n_elem);
  for (arma::uword j = 0; j < initial.n_elem; ++j) {
    bool low = std::isfinite(lo[j]), high = std::isfinite(hi[j]);
    bounds[j] = low ? (high ? 2 : 1) : (high ? 3 : 0);
  }
  OptimContext context{objective, arma::vec(), Evaluation{0, arma::vec()}, false};
  char message[60] = {};
  // These are the same native optimizer and default memory size as stats::optim.
  lbfgsb(initial.n_elem, 5, result.par.memptr(), lo.memptr(), hi.memptr(),
         bounds.data(), &result.value, objective_callback, gradient_callback,
         &result.code, &context, factr, pgtol, &result.fncount, &result.grcount,
         maxit, message, 0, 10);
  result.message = message;
  return result;
}

Geometry prepare_geometry(const std::vector<arma::vec>& times) {
  Geometry geometry;
  geometry.times = times;
  geometry.counts.set_size(times.size());
  std::vector<double> gaps;
  std::vector<arma::mat> differences;
  for (std::size_t i = 0; i < times.size(); ++i) {
    arma::uword m = times[i].n_elem;
    geometry.counts[i] = m;
    arma::mat delta(m, m);
    for (arma::uword j = 0; j < m; ++j) {
      for (arma::uword k = 0; k < m; ++k) {
        delta(k, j) = std::abs(times[i][k] - times[i][j]);
        if (delta(k, j) > 0) gaps.push_back(delta(k, j));
        else if (j != k) geometry.repeated = true;
      }
    }
    differences.push_back(std::move(delta));
  }
  if (!gaps.empty()) {
    std::sort(gaps.begin(), gaps.end());
    geometry.has_gaps = true;
    geometry.gap = gaps.front();
    geometry.max_gap = gaps.back();
    std::size_t mid = gaps.size() / 2;
    geometry.median_gap = gaps.size() % 2 ? gaps[mid] : (gaps[mid - 1] + gaps[mid]) / 2;
    for (double scale : {geometry.gap, std::sqrt(geometry.gap * geometry.median_gap),
                         geometry.median_gap, std::sqrt(geometry.median_gap * geometry.max_gap),
                         geometry.max_gap}) {
      if (std::find(geometry.scales.begin(), geometry.scales.end(), scale) == geometry.scales.end())
        geometry.scales.push_back(scale);
    }
  }
  for (std::size_t i = 0; i < times.size(); ++i) {
    arma::mat dist2 = arma::square(differences[i] / geometry.gap);
    std::size_t group = 0;
    while (group < geometry.dist2.size()) {
      if (geometry.dist2[group].n_rows == dist2.n_rows &&
          arma::all(arma::vectorise(geometry.dist2[group] == dist2))) break;
      ++group;
    }
    if (group == geometry.dist2.size()) {
      geometry.dist2.push_back(dist2);
      geometry.positive.push_back(arma::find(dist2 > 0));
      geometry.log_dist2.push_back(arma::log(dist2.elem(geometry.positive.back())));
      arma::vec unique_times = arma::unique(times[i]);
      geometry.rank.push_back(unique_times.n_elem);
    }
    geometry.subject_group.push_back(group);
  }
  return geometry;
}

// Shared RBF geometry for likelihoods and posterior conditioning.
void rbf_kernel(const Geometry& geometry, std::size_t group, double log_length,
                arma::mat& correlation, arma::mat* derivative) {
  const arma::uword dimension = geometry.dist2[group].n_rows;
  correlation.ones(dimension, dimension);
  if (derivative != nullptr) derivative->zeros(dimension, dimension);
  const arma::uvec& positive = geometry.positive[group];
  if (std::isfinite(log_length)) {
    const arma::vec log_ratio = geometry.log_dist2[group] - 2 * log_length - std::log(2.0);
    const arma::vec ratio = arma::exp(log_ratio);
    correlation.elem(positive) = arma::exp(-ratio);
    if (derivative != nullptr) {
      derivative->elem(positive) = 2 * arma::exp(log_ratio - ratio);
    }
  } else if (log_length < 0) {
    correlation.elem(positive).zeros();
  }
}

PriorResult gaussian_update(const arma::vec& x, double s) {
  PriorResult fit;
  fit.mean.zeros(x.n_elem);
  fit.second.zeros(x.n_elem);
  if (std::isinf(s)) {
    fit.par = Rcpp::NumericVector::create(Rcpp::_["mu"] = 0, Rcpp::_["eta"] = 0);
    return fit;
  }
  double s2 = s * s;
  double variance = std::max(arma::mean(arma::square(x)) - s2, 0.0);
  fit.par = Rcpp::NumericVector::create(Rcpp::_["mu"] = 0, Rcpp::_["eta"] = std::sqrt(variance));
  fit.mean = variance * x / (variance + s2);
  fit.second = arma::square(fit.mean) + variance * s2 / (variance + s2);
  fit.loglik = 0;
  for (double value : x) fit.loglik += R::dnorm(value, 0, std::sqrt(s2 + variance), true);
  fit.neg_kl = fit.loglik + 0.5 *
    (x.n_elem * std::log(2 * arma::datum::pi * s2) +
     arma::accu(arma::square(x) + fit.second - 2 * x % fit.mean) / s2);
  return fit;
}

Rcpp::List prior_output(const PriorResult& fit, const std::string& prior) {
  Rcpp::List output = Rcpp::List::create(
    Rcpp::_["prior"] = prior, Rcpp::_["par"] = fit.par,
    Rcpp::_["loglik"] = fit.loglik, Rcpp::_["neg_KL"] = fit.neg_kl);
  if (fit.extra.size()) {
    Rcpp::CharacterVector names = fit.extra.names();
    for (int j = 0; j < fit.extra.size(); ++j)
      output[Rcpp::as<std::string>(names[j])] = fit.extra[j];
  }
  return output;
}

Factors empty_factors(arma::uword variables, arma::uword visits, arma::uword capacity) {
  Factors f;
  f.U.zeros(variables, capacity);
  f.U2.zeros(variables, capacity);
  f.V.zeros(visits, capacity);
  f.V2.zeros(visits, capacity);
  f.u_loglik.zeros(capacity);
  f.v_loglik.zeros(capacity);
  f.u_kl.zeros(capacity);
  f.v_kl.zeros(capacity);
  f.u_par.resize(capacity);
  f.v_par.resize(capacity);
  f.v_diagnostics.resize(capacity);
  f.v_converged.resize(capacity, NA_LOGICAL);
  return f;
}

Factors initialize_factors(const WorkingData& data, arma::uword rank) {
  Factors f = empty_factors(data.X.n_rows, data.X.n_cols, rank);
  f.rank = rank;
  if (rank == 0) return f;
  arma::mat unfolded = data.X;
  if (data.aligned) {
    // Preserve the original time-major SVD, despite subject-major working storage.
    for (arma::uword m = 0; m < data.visits; ++m)
      for (arma::uword i = 0; i < data.subjects; ++i)
        unfolded.col(m * data.subjects + i) = data.X.col(i * data.visits + m);
  }
  arma::mat u, v;
  arma::vec singular;
  if (rank == 1 && std::min(unfolded.n_rows, unfolded.n_cols) >= 128) {
    SvdGramOperator op(unfolded);
    if (op.scale == 0) return f;
    arma::newarp::SymEigsSolver<double, arma::newarp::EigsSelect::LARGEST_ALGE, SvdGramOperator> solver(op, 1, 20);
    solver.init();
    if (solver.compute(1000, 1e-12) != 1)
      Rcpp::stop("Leading-component SVD did not converge.");
    singular.set_size(1);
    if (op.right) {
      v = solver.eigenvectors(1);
      u = unfolded * v;
      singular[0] = arma::norm(u.col(0));
      u /= singular[0];
    } else {
      u = solver.eigenvectors(1);
      v = unfolded.t() * u;
      singular[0] = arma::norm(v.col(0));
      v /= singular[0];
    }
    if (arma::norm(unfolded * v - singular[0] * u, "fro") > 1e-8 * singular[0] ||
        arma::norm(unfolded.t() * u - singular[0] * v, "fro") > 1e-8 * singular[0])
      Rcpp::stop("Leading-component SVD residual exceeds tolerance.");
  } else if (!arma::svd_econ(u, singular, v, unfolded, "both", "dc")) {
    Rcpp::stop("Dense SVD initialization failed.");
  }
  for (arma::uword k = 0; k < rank; ++k) {
    double scale = std::sqrt(singular[k]);
    f.U.col(k) = u.col(k) * scale;
    f.U2.col(k) = arma::square(u.col(k)) * (scale * scale);
    for (arma::uword t = 0; t < data.X.n_cols; ++t) {
      arma::uword index = data.aligned ? (t % data.visits) * data.subjects + t / data.visits : t;
      f.V(t, k) = v(index, k) * scale;
      f.V2(t, k) = v(index, k) * v(index, k) * (scale * scale);
    }
  }
  return f;
}

ResidualCache initialize_residual(const WorkingData& data, const Factors& f) {
  ResidualCache cache;
  cache.residual = data.X;
  if (f.rank) cache.residual -= f.U.head_cols(f.rank) * f.V.head_cols(f.rank).t();
  cache.squared_error = arma::sum(arma::square(cache.residual), 0).t();
  cache.uncertainty.zeros(data.X.n_cols);
  cache.rss.zeros(data.group_size.n_elem);
  for (arma::uword k = 0; k < f.rank; ++k) {
    double u_square = arma::dot(f.U.col(k), f.U.col(k));
    double u_second = arma::accu(f.U2.col(k));
    cache.uncertainty += (u_second - u_square) * f.V2.col(k) +
      u_square * (f.V2.col(k) - arma::square(f.V.col(k)));
  }
  for (arma::uword t = 0; t < data.X.n_cols; ++t)
    cache.rss[data.noise_group[t]] += cache.squared_error[t] + cache.uncertainty[t];
  return cache;
}

void replace_loading(const WorkingData& data, Factors& f, ResidualCache& cache,
                     arma::uword k, const PriorResult& update) {
  arma::vec delta = f.U.col(k) - update.mean;
  double old_square = arma::dot(f.U.col(k), f.U.col(k));
  double old_second = arma::accu(f.U2.col(k));
  double new_square = arma::dot(update.mean, update.mean);
  double new_second = arma::accu(update.second);
  for (arma::uword t = 0; t < data.X.n_cols; ++t) {
    cache.residual.col(t) += delta * f.V(t, k);
    double next_error = arma::dot(cache.residual.col(t), cache.residual.col(t));
    double v_square = f.V(t, k) * f.V(t, k);
    double change = ((new_second - new_square) - (old_second - old_square)) * f.V2(t, k) +
      (new_square - old_square) * (f.V2(t, k) - v_square);
    cache.rss[data.noise_group[t]] += next_error - cache.squared_error[t] + change;
    cache.squared_error[t] = next_error;
    cache.uncertainty[t] += change;
  }
  f.U.col(k) = update.mean;
  f.U2.col(k) = update.second;
  f.u_par[k] = update.par;
  f.u_loglik[k] = update.loglik;
  f.u_kl[k] = update.neg_kl;
  f.u_count = std::max(f.u_count, k + 1);
}

void replace_score(const WorkingData& data, Factors& f, ResidualCache& cache,
                   arma::uword k, const PriorResult& update) {
  double u_square = arma::dot(f.U.col(k), f.U.col(k));
  double u_second = arma::accu(f.U2.col(k));
  for (arma::uword t = 0; t < data.X.n_cols; ++t) {
    cache.residual.col(t) += f.U.col(k) * (f.V(t, k) - update.mean[t]);
    double next_error = arma::dot(cache.residual.col(t), cache.residual.col(t));
    double old_square = f.V(t, k) * f.V(t, k);
    double new_square = update.mean[t] * update.mean[t];
    double change = (u_second - u_square) * (update.second[t] - f.V2(t, k)) +
      u_square * ((update.second[t] - new_square) - (f.V2(t, k) - old_square));
    cache.rss[data.noise_group[t]] += next_error - cache.squared_error[t] + change;
    cache.squared_error[t] = next_error;
    cache.uncertainty[t] += change;
  }
  f.V.col(k) = update.mean;
  f.V2.col(k) = update.second;
  f.v_par[k] = update.par;
  f.v_loglik[k] = update.loglik;
  f.v_kl[k] = update.neg_kl;
  f.v_count = std::max(f.v_count, k + 1);
  if (update.extra.containsElementNamed("optimization")) {
    f.v_diagnostics[k] = update.extra["optimization"];
    f.diagnostics = true;
  } else {
    f.v_diagnostics[k] = R_NilValue;
  }
  if (update.extra.containsElementNamed("converged")) {
    f.v_converged[k] = Rcpp::as<int>(update.extra["converged"]);
    f.diagnostics = true;
  }
}

arma::vec pooled_noise_precision(const WorkingData& data, const ResidualCache& cache,
                                const arma::vec& previous) {
  const arma::uvec observed = arma::find(data.observed_count > 0);
  const arma::uvec empty = arma::find(data.observed_count == 0);
  const arma::vec counts = data.observed_count.elem(observed);
  const arma::vec rss = cache.rss.elem(observed);
  const double total = arma::accu(counts);
  const arma::vec weights = counts / total;
  const double missing_rss = arma::accu(cache.rss.elem(empty));
  const double scale = arma::accu(cache.rss) / total;
  // Unobserved visits share the observation-count-weighted mean variance.
  // Optimize the observed variances jointly because that constraint couples
  // their derivatives to the missing visits' posterior signal uncertainty.
  Objective objective = [&](const arma::vec& log_variance) {
    const arma::vec variance = arma::exp(log_variance);
    const double pooled = arma::dot(weights, variance);
    Evaluation value;
    value.value = (arma::dot(counts, log_variance) +
      arma::accu(rss / (scale * variance)) + missing_rss / (scale * pooled)) / (2 * total);
    value.gradient = (counts - rss / (scale * variance) -
      missing_rss / (scale * pooled * pooled) * weights % variance) / (2 * total);
    return value;
  };
  const arma::vec initial = arma::log(1 / (previous.elem(observed) * scale));
  const OptimResult optimum = optimize_lbfgsb(initial,
    arma::vec(observed.n_elem, arma::fill::value(-arma::datum::inf)),
    arma::vec(observed.n_elem, arma::fill::value(arma::datum::inf)), objective, 500, 100, 1e-8);
  if (optimum.code != 0) Rcpp::stop("Pooled noise variance optimization did not converge.");
  const arma::vec variance = scale * arma::exp(optimum.par);
  arma::vec precision(data.observed_count.n_elem);
  precision.elem(observed) = 1 / variance;
  precision.elem(empty).fill(1 / arma::dot(weights, variance));
  return precision;
}

double evidence_lower_bound(const WorkingData& data, const Factors& f,
                            const ResidualCache& cache, const arma::vec& tau) {
  if (f.u_count < f.rank || f.v_count < f.rank) return R_NegInf;
  double value = 0;
  for (arma::uword j = 0; j < tau.n_elem; ++j) {
    // Profiling q(Y_missing)=N(fitted mean, 1/tau) cancels the missing
    // Gaussian normalizers with its entropy. Signal uncertainty remains.
    value -= data.observed_count[j] / 2 * std::log(2 * arma::datum::pi / tau[j]);
    value -= tau[j] / 2 * cache.rss[j];
  }
  return value + arma::accu(f.u_kl.head(f.rank)) + arma::accu(f.v_kl.head(f.rank));
}
} // namespace evblm
