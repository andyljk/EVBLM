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
```

`fn = "Free"` is also available for aligned data. `method = "single"` fits one
factor; `method = "greedy"` returns greedy estimation without backfitting.
`mvEBNM()` and `irrEBNM()` expose the covariance updates for direct inspection.
See `?evblm` and `?mvEBNM` for arguments and returned moment arrays.

## Source organization

- `src/engine.cpp`: factor selection, coordinate updates, convergence, and imputation.
- `src/aligned.cpp`: aligned EC, Free, and RBF covariance estimation.
- `src/irregular.cpp`: irregular EC and RBF covariance estimation and conditioning.
- `src/helper.cpp` and `src/helper.h`: shared optimizer, time geometry, RBF kernel,
  Gaussian loading update, dense SVD initialization, residual buffers, and ELBO summaries.
- `src/bindings.cpp`: covariance entry points exposed to R.
- `R/evblm.R`: validation and conversion between public arrays/lists and native matrices.
- `tests/regression.R`: portable comparisons against fixtures from the original R code.

All numerical fitting, including optimizer iterations, runs in C++. The optimizer
is R's native L-BFGS-B routine, called directly without R function callbacks.
Initialization still uses the dense divide-and-conquer SVD; no truncated or
randomized initialization was introduced.

## Numerical compatibility

The reference implementations are the two scripts in `CSDA_Submission` from
September 2026. The port preserves their update order, prior boundaries, covariance
optimizer starts and tolerances, and completed-data imputation objective.
Comparisons use numerical tolerances because operation ordering and eigensolvers
can change rounding and the selected start among numerically tied optima.

Several original conventions are retained explicitly:

- Aligned fits can run `max_iter + 1` sweeps; irregular fits can run `max_iter`.
- Aligned imputation stops on the sum of squared changes at missing entries;
  irregular imputation uses the mean squared change across all entries.
- The initial ELBO is negative infinity until all point-mass initial factors update.
- Irregular single-factor and greedy fitting retain their progress printing.

Malformed inputs and all-zero or wholly missing data are rejected by the R interface.
Greedy selection is bounded by its existing factor-storage capacity; the original
aligned script could instead overrun that capacity. Empty dimensions are rejected,
and zero-factor backfitting is supported.

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
