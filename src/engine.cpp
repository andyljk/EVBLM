#include "helper.h"
#include <algorithm>
#include <cmath>

namespace evblm {
namespace {

struct Trace {
  std::vector<int> iteration, index;
  std::vector<std::string> block;
  std::vector<double> elbo;

  void reserve(std::size_t size) {
    iteration.reserve(size);
    index.reserve(size);
    block.reserve(size);
    elbo.reserve(size);
  }
  void append(int step, const std::string& name, int coordinate, double value) {
    iteration.push_back(step);
    block.push_back(name);
    index.push_back(coordinate);
    elbo.push_back(value);
  }
  Rcpp::DataFrame output() const {
    Rcpp::RObject coordinates;
    if (std::all_of(index.begin(), index.end(), [](int value) { return value == NA_INTEGER; }))
      coordinates = Rcpp::LogicalVector(index.size(), NA_LOGICAL);
    else coordinates = Rcpp::wrap(index);
    return Rcpp::DataFrame::create(Rcpp::_["iteration"] = iteration,
      Rcpp::_["block"] = block, Rcpp::_["index"] = coordinates, Rcpp::_["elbo"] = elbo);
  }
};

struct Fit {
  Factors factors;
  arma::mat signal;
  arma::vec tau;
  std::vector<double> elbo;
  Trace trace;
  int iterations = 0;
};

Fit iterate(WorkingData data, Factors factors, const std::string& prior,
            bool impute, double threshold, int max_iter, bool verbose,
            bool single) {
  Fit fit;
  fit.factors = std::move(factors);
  Factors& f = fit.factors;
  ResidualCache cache = initialize_residual(data, f);
  fit.signal = data.X - cache.residual;
  fit.tau.ones(data.group_size.n_elem);
  if (!data.aligned) {
    double rss = f.u_count == f.rank && f.v_count == f.rank ?
      cache.rss[0] : arma::accu(arma::square(data.X));
    fit.tau[0] = data.X.n_elem / rss;
  }
  int limit = max_iter + (data.aligned ? 1 : 0);
  fit.elbo.reserve(limit);
  fit.trace.reserve(1 + static_cast<std::size_t>(limit) *
    (data.group_size.n_elem + 2 * f.rank + (impute ? 1 : 0)));
  fit.trace.append(0, "initial", NA_INTEGER,
                   evidence_lower_bound(data, f, cache, fit.tau));
  double change = impute && data.aligned && data.missing.is_empty() ? 0 : R_PosInf;

  while (change > threshold && fit.iterations < limit) {
    Rcpp::checkUserInterrupt();
    arma::mat previous_signal = fit.signal;
    int iteration = fit.iterations + 1;
    for (arma::uword j = 0; j < fit.tau.n_elem; ++j) {
      if (data.aligned || (f.u_count == f.rank && f.v_count == f.rank))
        fit.tau[j] = data.X.n_rows * data.group_size[j] / cache.rss[j];
      fit.trace.append(iteration, "noise", data.aligned ? j + 1 : NA_INTEGER,
                       evidence_lower_bound(data, f, cache, fit.tau));
    }
    arma::vec weights(data.X.n_cols);
    for (arma::uword t = 0; t < weights.n_elem; ++t)
      weights[t] = fit.tau[data.noise_group[t]];

    for (arma::uword k = 0; k < f.rank; ++k) {
      // Add back only factor k. No full leave-one-factor-out array is allocated.
      arma::vec weighted_scores = weights % f.V.col(k);
      double precision = arma::dot(weights, f.V2.col(k));
      arma::vec pseudo_u(data.X.n_rows, arma::fill::zeros);
      if (precision != 0) {
        pseudo_u = (cache.residual * weighted_scores +
          f.U.col(k) * arma::dot(f.V.col(k), weighted_scores)) / precision;
      }
      PriorResult loading = gaussian_update(pseudo_u, std::pow(precision, -0.5));
      replace_loading(data, f, cache, k, loading);
      fit.trace.append(iteration, "u", k + 1, evidence_lower_bound(data, f, cache, fit.tau));

      double second_sum = arma::accu(f.U2.col(k));
      arma::vec pseudo_v(data.X.n_cols, arma::fill::zeros);
      if (second_sum != 0) {
        pseudo_v = (cache.residual.t() * f.U.col(k) +
          arma::dot(f.U.col(k), f.U.col(k)) * f.V.col(k)) / second_sum;
      }
      PriorResult scores;
      if (data.aligned) {
        arma::mat pseudo(data.subjects, data.visits);
        for (arma::uword i = 0; i < data.subjects; ++i)
          for (arma::uword m = 0; m < data.visits; ++m)
            pseudo(i, m) = pseudo_v[i * data.visits + m];
        arma::vec standard_error = 1 / arma::sqrt(second_sum * fit.tau);
        PriorResult aligned_scores = fit_aligned(pseudo, standard_error, data.geometry, prior, f.v_par[k]);
        scores = aligned_scores;
        for (arma::uword i = 0; i < data.subjects; ++i) {
          for (arma::uword m = 0; m < data.visits; ++m) {
            scores.mean[i * data.visits + m] = aligned_scores.mean[i + data.subjects * m];
            scores.second[i * data.visits + m] = aligned_scores.second[i + data.subjects * m];
          }
        }
      } else {
        std::vector<arma::vec> pseudo;
        pseudo.reserve(data.subjects);
        for (arma::uword i = 0; i < data.subjects; ++i)
          pseudo.push_back(pseudo_v.subvec(data.offsets[i], data.offsets[i + 1] - 1));
        scores = fit_irregular(pseudo, std::pow(second_sum * fit.tau[0], -0.5),
                               data.geometry, prior, f.v_par[k]);
      }
      replace_score(data, f, cache, k, scores);
      fit.trace.append(iteration, "v", k + 1, evidence_lower_bound(data, f, cache, fit.tau));
    }
    fit.signal = data.X - cache.residual;
    if (impute) {
      // The original objective uses completed data; missing values change only here.
      for (arma::uword index : data.missing) {
        arma::uword column = index / data.X.n_rows;
        double squared = cache.residual[index] * cache.residual[index];
        data.X[index] = fit.signal[index];
        cache.residual[index] = 0;
        cache.squared_error[column] -= squared;
        cache.rss[data.noise_group[column]] -= squared;
      }
      fit.trace.append(iteration, "impute", NA_INTEGER,
                       evidence_lower_bound(data, f, cache, fit.tau));
    }
    fit.elbo.push_back(evidence_lower_bound(data, f, cache, fit.tau));
    arma::mat difference = fit.signal - previous_signal;
    // Preserve the original aligned imputation sum-of-squares stopping rule.
    change = impute && data.aligned ? arma::accu(arma::square(difference.elem(data.missing))) :
      arma::accu(arma::square(difference)) / difference.n_elem;
    ++fit.iterations;
    if (verbose || (single && !data.aligned)) Rprintf("[1] %.7g\n", change);
  }
  return fit;
}

Fit greedy(const WorkingData& data, const std::string& prior, double threshold) {
  Fit fit;
  arma::uword capacity = std::min(data.X.n_rows, data.aligned ? data.subjects : data.X.n_cols);
  fit.factors = empty_factors(data.X.n_rows, data.X.n_cols, capacity);
  Factors& f = fit.factors;
  ResidualCache cache = initialize_residual(data, f);
  fit.signal.zeros(data.X.n_rows, data.X.n_cols);
  fit.tau = data.X.n_rows * data.group_size / cache.rss;
  fit.elbo.reserve(capacity);
  while (f.rank < capacity) {
    WorkingData residual_data = data;
    residual_data.X = cache.residual;
    residual_data.missing.reset();
    Fit candidate = iterate(residual_data, initialize_factors(residual_data, 1),
                             prior, false, threshold, 50, false, true);
    if (arma::all(arma::vectorise(candidate.factors.U2) == 0) ||
        arma::all(arma::vectorise(candidate.factors.V2) == 0)) break;
    arma::uword k = f.rank++;
    PriorResult loading, scores;
    loading.mean = candidate.factors.U.col(0);
    loading.second = candidate.factors.U2.col(0);
    loading.par = candidate.factors.u_par[0];
    loading.loglik = candidate.factors.u_loglik[0];
    loading.neg_kl = candidate.factors.u_kl[0];
    scores.mean = candidate.factors.V.col(0);
    scores.second = candidate.factors.V2.col(0);
    scores.par = candidate.factors.v_par[0];
    scores.loglik = candidate.factors.v_loglik[0];
    scores.neg_kl = candidate.factors.v_kl[0];
    replace_loading(data, f, cache, k, loading);
    replace_score(data, f, cache, k, scores);
    double objective = evidence_lower_bound(data, f, cache, candidate.tau);
    if (k > 0 && objective <= fit.elbo.back()) {
      // Undo the rejected component; signal and noise retain the last accepted fit.
      PriorResult zero;
      zero.mean.zeros(data.X.n_rows);
      zero.second.zeros(data.X.n_rows);
      replace_loading(data, f, cache, k, zero);
      zero.mean.zeros(data.X.n_cols);
      zero.second.zeros(data.X.n_cols);
      replace_score(data, f, cache, k, zero);
      --f.rank;
      f.u_count = f.v_count = f.rank;
      break;
    }
    fit.tau = candidate.tau;
    fit.elbo.push_back(objective);
    fit.signal = data.X - cache.residual;
    if (!data.aligned) Rprintf("[1] %d\n", static_cast<int>(f.rank));
  }
  return fit;
}

void remove_null_factors(const WorkingData& data, Factors& f) {
  std::vector<arma::uword> keep;
  bool score_null = !data.aligned;
  for (arma::uword k = 0; k < f.rank; ++k) {
    if (arma::accu(arma::abs(f.U.col(k))) >= 1e-8) keep.push_back(k);
    if (data.aligned) {
      for (arma::uword m = 0; m < data.visits; ++m) {
        double total = 0;
        for (arma::uword i = 0; i < data.subjects; ++i) total += std::abs(f.V(i * data.visits + m, k));
        if (total < 1e-8) score_null = true;
      }
    }
  }
  if (!score_null || keep.size() == f.rank) return;
  Factors reduced = empty_factors(data.X.n_rows, data.X.n_cols, keep.size());
  reduced.rank = keep.size();
  reduced.diagnostics = f.diagnostics;
  for (std::size_t j = 0; j < keep.size(); ++j) {
    arma::uword k = keep[j];
    reduced.U.col(j) = f.U.col(k);
    reduced.U2.col(j) = f.U2.col(k);
    reduced.V.col(j) = f.V.col(k);
    reduced.V2.col(j) = f.V2.col(k);
    reduced.u_loglik[j] = f.u_loglik[k];
    reduced.v_loglik[j] = f.v_loglik[k];
    reduced.u_kl[j] = f.u_kl[k];
    reduced.v_kl[j] = f.v_kl[k];
    reduced.u_par[j] = f.u_par[k];
    reduced.v_par[j] = f.v_par[k];
    reduced.v_diagnostics[j] = f.v_diagnostics[k];
    reduced.v_converged[j] = f.v_converged[k];
    if (k < f.u_count) ++reduced.u_count;
    if (k < f.v_count) ++reduced.v_count;
  }
  f = std::move(reduced);
}

Rcpp::List output_fit(const Fit& fit, bool aligned, bool is_greedy) {
  const Factors& f = fit.factors;
  Rcpp::List upar(f.u_count), vpar(f.v_count);
  for (arma::uword k = 0; k < f.u_count; ++k) upar[k] = f.u_par[k];
  for (arma::uword k = 0; k < f.v_count; ++k) vpar[k] = f.v_par[k];
  Rcpp::List u = Rcpp::List::create(Rcpp::_["mean"] = arma::mat(f.U.head_cols(f.rank)),
    Rcpp::_["second"] = arma::mat(f.U2.head_cols(f.rank)), Rcpp::_["par"] = upar);
  Rcpp::List v = Rcpp::List::create(Rcpp::_["mean"] = arma::mat(f.V.head_cols(f.rank)),
    Rcpp::_["second"] = arma::mat(f.V2.head_cols(f.rank)), Rcpp::_["par"] = vpar);
  if (f.u_count || (f.rank == 0 && !is_greedy)) {
    u["loglik"] = Rcpp::wrap(std::vector<double>(f.u_loglik.begin(), f.u_loglik.begin() + f.u_count));
    u["neg_KL"] = Rcpp::wrap(std::vector<double>(f.u_kl.begin(), f.u_kl.begin() + f.u_count));
  }
  if (f.v_count || (f.rank == 0 && !is_greedy)) {
    v["loglik"] = Rcpp::wrap(std::vector<double>(f.v_loglik.begin(), f.v_loglik.begin() + f.v_count));
    v["neg_KL"] = Rcpp::wrap(std::vector<double>(f.v_kl.begin(), f.v_kl.begin() + f.v_count));
  }
  if (!aligned && f.diagnostics && !is_greedy) {
    Rcpp::List diagnostics(f.v_count);
    Rcpp::LogicalVector converged(f.v_count);
    for (arma::uword k = 0; k < f.v_count; ++k) {
      diagnostics[k] = f.v_diagnostics[k];
      converged[k] = f.v_converged[k];
    }
    v["optimization"] = diagnostics;
    v["converged"] = converged;
  }
  Rcpp::RObject elbo = fit.elbo.empty() ? R_NilValue : Rcpp::wrap(fit.elbo);
  Rcpp::List output = Rcpp::List::create(Rcpp::_["u"] = u, Rcpp::_["v"] = v,
    Rcpp::_["elbo"] = elbo, Rcpp::_["S"] = fit.signal,
    Rcpp::_["noise"] = Rcpp::wrap(std::vector<double>(fit.tau.begin(), fit.tau.end())),
    Rcpp::_["n_factors"] = static_cast<int>(f.rank), Rcpp::_["n_iter"] = fit.iterations);
  if (!is_greedy) output["elbo_steps"] = fit.trace.output();
  return output;
}
} // namespace
} // namespace evblm

// [[Rcpp::export]]
Rcpp::List evblm_engine_cpp(const arma::mat& X, const Rcpp::List& D, bool aligned,
                            int subjects, const std::string& prior, const std::string& method,
                            int rank, bool impute, double thres, int max_iter,
                            bool verbose, bool null_check,
                            Rcpp::Nullable<Rcpp::List> initial_u = R_NilValue,
                            Rcpp::Nullable<Rcpp::List> initial_v = R_NilValue) {
  using namespace evblm;
  WorkingData data;
  data.X = X;
  data.aligned = aligned;
  data.subjects = subjects;
  std::vector<arma::vec> times;
  for (int i = 0; i < D.size(); ++i) times.push_back(Rcpp::as<arma::vec>(D[i]));
  data.geometry = prepare_geometry(times);
  data.visits = aligned ? times[0].n_elem : 0;
  data.offsets.zeros(subjects + 1);
  data.noise_group.zeros(X.n_cols);
  data.group_size.zeros(aligned ? data.visits : 1);
  for (int i = 0; i < subjects; ++i) {
    arma::uword count = aligned ? data.visits : times[i].n_elem;
    data.offsets[i + 1] = data.offsets[i] + count;
    for (arma::uword m = 0; m < count; ++m) {
      arma::uword group = aligned ? m : 0;
      data.noise_group[data.offsets[i] + m] = group;
      data.group_size[group] += 1;
    }
  }
  data.missing = arma::find_nonfinite(arma::vectorise(data.X));
  if (impute) data.X.elem(data.missing).zeros();
  Fit fit;
  if (method == "greedy" || method == "greedy+backfit") {
    fit = greedy(data, prior, thres);
    if (method == "greedy+backfit")
      fit = iterate(data, std::move(fit.factors), prior, impute, thres, max_iter, verbose, false);
  } else {
    Factors factors;
    if (initial_u.isNotNull() && initial_v.isNotNull()) {
      Rcpp::List u(initial_u), v(initial_v);
      factors = empty_factors(X.n_rows, X.n_cols, rank);
      factors.rank = rank;
      factors.U = Rcpp::as<arma::mat>(u["mean"]);
      factors.U2 = Rcpp::as<arma::mat>(u["second"]);
      factors.V = Rcpp::as<arma::mat>(v["mean"]);
      factors.V2 = Rcpp::as<arma::mat>(v["second"]);
      Rcpp::List up = u["par"], vp = v["par"];
      for (int k = 0; k < up.size(); ++k) factors.u_par[k] = up[k];
      for (int k = 0; k < vp.size(); ++k) factors.v_par[k] = vp[k];
      if (u.containsElementNamed("neg_KL")) {
        arma::vec negative_kl = Rcpp::as<arma::vec>(u["neg_KL"]);
        if (!negative_kl.is_empty()) {
          factors.u_loglik = Rcpp::as<arma::vec>(u["loglik"]);
          factors.u_kl = negative_kl;
          factors.u_count = negative_kl.n_elem;
        }
      }
      if (v.containsElementNamed("neg_KL")) {
        arma::vec negative_kl = Rcpp::as<arma::vec>(v["neg_KL"]);
        if (!negative_kl.is_empty()) {
          factors.v_loglik = Rcpp::as<arma::vec>(v["loglik"]);
          factors.v_kl = negative_kl;
          factors.v_count = negative_kl.n_elem;
        }
      }
      if (v.containsElementNamed("optimization")) {
        Rcpp::List diagnostics = v["optimization"];
        Rcpp::LogicalVector converged = v["converged"];
        for (int k = 0; k < diagnostics.size(); ++k) {
          factors.v_diagnostics[k] = diagnostics[k];
          factors.v_converged[k] = converged[k];
        }
        factors.diagnostics = true;
      }
    } else {
      factors = initialize_factors(data, method == "single" ? 1 : rank);
    }
    fit = iterate(data, std::move(factors), prior, impute, thres, max_iter, verbose, method == "single");
  }
  if (null_check && method != "single" && method != "greedy") remove_null_factors(data, fit.factors);
  return output_fit(fit, aligned, method == "greedy");
}
