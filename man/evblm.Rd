\name{evblm}
\alias{evblm}
\alias{evblm_impute}
\title{Fit an Empirical Variational Bayes Longitudinal Matrix Model}
\description{
Estimate variable loadings and subject scores from aligned or irregular
longitudinal data. The R interface validates and formats inputs and outputs;
initialization, covariance fitting, coordinate updates, factor selection,
objective tracking, and imputation run in C++. Optional non-normal loading
updates call \pkg{ebnm} from the fitting loop.
}
\usage{
evblm(X, D, R = NULL, fn = "EC",
  method = if (is.null(R)) "greedy+backfit" else "backfit",
  impute = FALSE, thres = 1e-8, max_iter = 50,
  verbose = FALSE, null_check = TRUE,
  u_mt = NULL, v_mt = NULL, D_init = NULL, loading_prior = "normal")

evblm_impute(X, D, R = NULL, fn = "EC",
  method = if (is.null(R)) "greedy+backfit" else "backfit",
  thres = 1e-5, max_iter = 100, verbose = FALSE,
  null_check = TRUE, D_init = NULL, loading_prior = "normal")
}
\arguments{
  \item{X}{Aligned data: a numeric array with dimensions variables by subjects
  by visits. Irregular data: a list of numeric variables by visits matrices,
  one per subject. Use \code{evblm()} for complete data and \code{evblm_impute()}
  for data with missing entries. A mismatch stops with a message directing the
  user to the appropriate function.}
  \item{D}{Aligned data: a finite numeric time vector. Irregular data: one
  finite numeric time vector per subject. Repeated visit times are allowed.}
  \item{R}{Nonnegative number of factors for \code{method = "backfit"}; zero
  gives a noise-only fit. The single-factor
  method uses one factor. Greedy methods select factors sequentially.}
  \item{fn}{Covariance model: \code{"EC"} (exchangeable), \code{"RBF"}
  (radial basis), or, for aligned data, \code{"Free"} (unrestricted).}
  \item{method}{\code{"single"}, \code{"backfit"}, \code{"greedy"}, or
  \code{"greedy+backfit"}. Imputation supports the latter combined method
  and \code{"backfit"}.}
  \item{impute}{Whether to iteratively replace missing entries with fitted
  signal values. \code{evblm_impute()} sets this to \code{TRUE}.
  Imputation requires at least one missing entry.}
  \item{thres}{Nonnegative convergence threshold for changes in the fitted signal.
  Imputation uses the sum of squared changes at missing entries, as in Algorithm 4;
  complete-data fitting uses the mean squared change over all entries.}
  \item{max_iter}{Nonnegative iteration limit. The aligned implementation
  preserves its inclusive iteration condition and can perform
  \code{max_iter + 1} sweeps. Greedy factor proposals retain the original
  single-factor iteration limit.}
  \item{verbose}{Whether to print iteration diagnostics.}
  \item{null_check}{Whether to remove factors whose loading posterior means
  have an absolute column sum below \code{1e-8}.}
  \item{u_mt, v_mt}{Optional initial factor lists from a compatible fit.
  Supply both for non-imputing backfitting. Their \code{pos} fields store
  posterior means and second moments in the shapes described below.
  If score optimization metadata are supplied, include both
  \code{optimization} and \code{converged}, with one entry per factor.}
  \item{D_init}{Optional irregular initialization schedules, retained for
  compatibility. They must have the same subject and visit counts as \code{D}.
  Initialization uses their block sizes, not alternate visit times.}
  \item{loading_prior}{Loading prior family: \code{"normal"} (the default),
  \code{"point_normal"}, \code{"point_laplace"},
  or \code{"normal_scale_mixture"}. All are centered
  at zero, with a separate prior estimated for each factor. Non-normal
  options require \pkg{ebnm}.}
}
\details{
Loading updates retain posterior means and second moments for every prior.
Spike-and-slab priors therefore do not generally produce exact zeros in the
posterior means. The default normal update retains the native closed-form
implementation. Other families use \code{ebnm::ebnm()} and its default
optimization settings, without an additional penalty on the prior.
For normal scale mixtures, the first update chooses the scale grid and later
updates estimate its weights using the previous fitted prior as a warm start.
Retaining the grid keeps the previous prior available to the next optimization.

If all score second moments vanish, the existing
zero-factor boundary is retained for all loading priors.

The aligned exchangeable model allows negative correlations within its
positive-semidefinite parameter space. The irregular exchangeable model has
nonnegative independent and random-intercept variance components.

The numerical implementation uses SVD initialization, with finite noise
initialization for exactly fitted aligned data before posterior moments are
available. It preserves the original update order and covariance optimizer
settings. Imputation uses the manuscript's missing-entry stopping criterion.
Single-factor initializations use a leading-component iterative SVD when both
matrix dimensions are at least 128; smaller and multi-factor initializations use
dense SVD. The iterative solver checks convergence without changing R's RNG state.
Residuals and objective summaries are maintained incrementally. Covariance
geometry is reused, and subjects with identical irregular time-distance
matrices share covariance decompositions. Numerical results can differ at
floating-point precision from an R implementation.

Imputation assigns independent Gaussian variational distributions to missing
values, with fitted means and the current noise variance. Their entropy removes
missing entries from the Gaussian normalization count in the ELBO. Noise variance
is the total residual second moment divided by the number of observed entries;
posterior signal uncertainty at missing entries is retained. Loading and score
updates are unchanged. An aligned visit with no observations shares the
observation-count-weighted mean variance of the observed visits, optimized jointly.
}
\value{
A list containing \code{u} (loading factors), \code{v} (score factors),
\code{elbo} (objective per sweep), \code{S} (fitted signal in the shape of
\code{X}), and \code{noise} (precision, one per aligned visit or one common
irregular precision). \code{n_factors} is returned by multi-factor fits;
\code{n_iter} is returned by the single-factor method.

Factor lists contain \code{par}, \code{pos}, \code{loglik}, and
\code{neg_KL}. The two moment entries are the mean and the second moment,
not the variance. Loading moments have dimensions variables by factors by
two. Aligned score moments have dimensions subjects by factors by two by
visits. Irregular score moments are a list of factors by visits by two arrays.
Irregular score factors additionally retain covariance optimization diagnostics.
The loading list also records \code{prior}. Its \code{par} entries are
mean/standard-deviation vectors for normal loadings, or \pkg{ebnm}
\code{fitted_g} objects for other priors. A non-normal factor collapsed at
zero score precision has \code{par = NULL}. Warm starts with likelihood and
KL metadata must use the same loading prior; older fits without a
\code{prior} field are treated as normal.

Except for the greedy-only method, \code{elbo_steps} records the initial
objective and each noise, loading, score, and, when relevant, imputation update.
}
\examples{
set.seed(1)
x = array(rnorm(12 * 5 * 3), c(12, 5, 3))
fit = evblm(x, D = c(0, 1, 3), R = 1, max_iter = 2)
dim(fit$S)

if (requireNamespace("ebnm", quietly = TRUE)) {
  sparse_fit = evblm(x, D = c(0, 1, 3), R = 1, max_iter = 2,
    loading_prior = "point_normal")
}

times = list(c(0, 1), c(0, 2, 4), c(1, 3))
x_irregular = lapply(times, function(d) matrix(rnorm(12 * length(d)), 12))
fit_irregular = evblm(x_irregular, times, R = 1, max_iter = 2)
}
