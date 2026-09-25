# EVBLM

Empirical variational Bayes longitudinal matrix factorization, with a C++ fitting
engine and R interfaces for input validation and output formatting.

From this project directory, install the package with `R CMD INSTALL .`.
Rcpp, RcppArmadillo, and a C++17 compiler are required to build it.

```r
library(EVBLM)

# Aligned: X has dimensions variables x subjects x visits.
fit = evblm(X, D, R = 3, fn = "RBF")

# Irregular: X and D are matching lists of subject matrices and visit times.
fit = evblm(X_list, D_list, R = 3, fn = "EC")

# Estimate the number of factors, then refine all factors jointly.
fit = evblm(X, D, fn = "RBF", method = "greedy+backfit")

# Missing entries are marked NA.
fit = evblm_impute(X, D, R = 3, fn = "RBF")

# Estimate a sparse prior for each loading vector (requires ebnm).
fit = evblm(X, D, R = 3, fn = "RBF", loading_prior = "point_normal")
```

`fn = "Free"` is also available for aligned data. `method = "single"` fits one
factor; `method = "greedy"` returns greedy estimation without backfitting.
`mvEBNM()` and `irrEBNM()` expose the covariance updates for direct inspection.
See `?evblm` and `?mvEBNM` for arguments and returned moment arrays.

`loading_prior` is available in both `evblm()` and `evblm_impute()`. Choices
are `"normal"` (the default), `"point_normal"`, `"point_laplace"`,
and `"normal_scale_mixture"`. All priors are centered at zero
and estimated separately for each loading vector. Fits continue to use posterior
means and second moments, so sparse priors do not generally yield exact-zero
loading estimates. `fit$u$prior` records the family, and `fit$u$par` stores the
Gaussian parameter vectors or the `ebnm` fitted-prior objects. Warm starts with
likelihood and KL metadata must use the same loading prior.
Normal scale mixtures retain each factor's initial scale grid and re-estimate
its weights at subsequent updates, so changing the grid cannot lower the ELBO.

The three new options require `ebnm`. The existing
zero-factor boundary is retained when score second moments vanish; in that case
a non-normal factor's stored prior is `NULL`.

## Source organization

- `src/engine.cpp`: factor selection, coordinate updates, convergence, and imputation.
- `src/aligned.cpp`: aligned EC, Free, and RBF covariance estimation.
- `src/irregular.cpp`: irregular EC and RBF covariance estimation and conditioning.
- `src/helper.cpp` and `src/helper.h`: shared optimizer, time geometry, RBF kernel,
  Gaussian loading update, SVD initialization, residual buffers, and ELBO summaries.
- `src/bindings.cpp`: covariance entry points exposed to R.
- `R/evblm.R`: validation and conversion between public arrays/lists and native matrices.
- `tests/regression.R`: portable comparisons against fixtures from the original R code.

Normal-loading fits run entirely in C++; other loading priors call `ebnm` in R
once per factor update. The covariance optimizer
is R's native L-BFGS-B routine, called directly without R function callbacks.
Single-factor initialization (including greedy proposals) uses restarted Lanczos
when both matrix dimensions are at least 128. It applies matrix-vector products
without constructing a Gram matrix or a full SVD. Small matrices and multi-factor
initializations retain dense divide-and-conquer SVD. The iterative solver checks
convergence and singular-triplet residuals, and its local generator leaves R's RNG
state unchanged. Signs and bases within tied leading singular subspaces can differ.

## Numerical compatibility

The reference implementations are the two scripts in `CSDA_Submission` from
September 2026. The port preserves their update order, prior boundaries, covariance
optimizer starts and tolerances. Imputation uses the Gaussian augmentation below
instead of treating filled values as observed data.
Comparisons use numerical tolerances because operation ordering and eigensolvers
can change rounding and the selected start among numerically tied optima.

The following fitting conventions apply:

- Aligned fits can run `max_iter + 1` sweeps; irregular fits can run `max_iter`.
- Aligned and irregular imputation stop on the sum of squared changes at missing
  entries, following Algorithm 4. This corrects the irregular R stopping rule.
- The initial ELBO is negative infinity until all point-mass initial factors update.
- Irregular single-factor and greedy fitting retain their progress printing.

Aligned noise initialization uses data energy when SVD residuals are at roundoff;
the first noise update waits until all factors have posterior moments. Both data
layouts remove null factors using the loading criterion alone. Covariance
eigendecomposition failures raise an explicit error.

Malformed inputs and all-zero or wholly missing data are rejected by the R interface.
Greedy selection is bounded by its existing factor-storage capacity; the original
aligned script could instead overrun that capacity. Empty dimensions are rejected,
and zero-factor backfitting is supported.

## Gaussian imputation and noise estimation

Each missing value has an independent variational distribution
`N(fitted mean, sigma²)`. Profiling its variance cancels its Gaussian normalizer
against its entropy. If `RSS` includes squared residuals of the filled means and
posterior signal variances over all entries, the bound is
`-0.5 * sum(n_observed * log(2*pi*sigma²) + RSS/sigma²) - KL`.
Thus each noise update is `sigma² = RSS / n_observed`. After updating the filled
means, missing entries contribute only posterior signal uncertainty to `RSS`.
Loading and score updates retain their existing form. This is a variational
bound for the observed data; it is not the fully collapsed observed-data update
that would also remove missing-entry signal uncertainty from factor updates.

A completely unobserved aligned visit shares the observation-count-weighted mean
noise variance of the observed visits. Those variances are optimized jointly to
respect that constraint and the same bound. Entirely missing datasets remain
invalid. The correction applies during greedy selection as well as backfitting.

## Local validation and timing

The development review compared 60 valid covariance/fitting cases against the R
references at tolerance `2e-5`, including all five model variants, warm starts,
imputation, zero factors, repeated times, and prior boundaries. An additional 100
random residual-update sequences agreed with independently computed dense moments
at tolerance `1e-12`.
`R CMD check --no-manual` completed with no errors, warnings, or notes on the
development machine (macOS arm64, R 4.5.2).

A local benchmark used 120 variables, 24 subjects, 3 factors, five aligned visits
or irregular schedules of four to six visits, and `max_iter = 4`. Medians of three
runs were:

| Model | Original R (s) | C++ package (s) |
| --- | ---: | ---: |
| Aligned EC | 0.106 | 0.002 |
| Aligned Free | 0.046 | 0.002 |
| Aligned RBF | 0.134 | 0.007 |
| Irregular EC | 0.123 | 0.004 |
| Irregular RBF | 2.313 | 0.038 |

These are small local benchmarks, not general speedup guarantees. Relative fitted
signal errors were at most `1.2e-9` in these cases. The comparison script is in
`benchmarks/compare-reference.R` and uses the original scripts as its baseline.

The maintainer name and email in `DESCRIPTION` are placeholders for local development.
