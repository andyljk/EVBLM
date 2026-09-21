\name{evblm}
\alias{evblm}
\alias{evblm_impute}
\title{Fit an Empirical Variational Bayes Longitudinal Matrix Model}
\description{
Estimate variable loadings and subject scores from aligned or irregular
longitudinal data. The R interface validates and formats inputs and outputs;
initialization, covariance fitting, coordinate updates, factor selection,
objective tracking, and imputation run in C++.
}
\usage{
evblm(X, D, R = NULL, fn = "EC",
  method = if (is.null(R)) "greedy+backfit" else "backfit",
  impute = FALSE, thres = 1e-8, max_iter = 50,
  verbose = FALSE, null_check = TRUE,
  u_mt = NULL, v_mt = NULL, D_init = NULL)

evblm_impute(X, D, R = NULL, fn = "EC",
  method = if (is.null(R)) "greedy+backfit" else "backfit",
  thres = 1e-5, max_iter = 100, verbose = FALSE,
  null_check = TRUE, D_init = NULL)
}
\arguments{
  \item{X}{Aligned data: a numeric array with dimensions variables by subjects
  by visits. Irregular data: a list of numeric variables by visits matrices,
  one per subject. Missing entries require \code{impute = TRUE}.}
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
  signal values. \code{evblm_impute()} sets this to \code{TRUE}.}
  \item{thres}{Nonnegative convergence threshold for changes in the fitted signal.}
  \item{max_iter}{Nonnegative iteration limit. The aligned implementation
  preserves its inclusive iteration condition and can perform
  \code{max_iter + 1} sweeps. Greedy factor proposals retain the original
  single-factor iteration limit.}
  \item{verbose}{Whether to print iteration diagnostics.}
  \item{null_check}{Whether to remove factors meeting the original
  implementation's null-factor criteria.}
  \item{u_mt, v_mt}{Optional initial factor lists from a compatible fit.
  Supply both for non-imputing backfitting. Their \code{pos} fields store
  posterior means and second moments in the shapes described below.
  If score optimization metadata are supplied, include both
  \code{optimization} and \code{converged}, with one entry per factor.}
  \item{D_init}{Optional irregular initialization schedules, retained for
  compatibility. They must have the same subject and visit counts as \code{D}.
  Initialization uses their block sizes, not alternate visit times.}
}
\details{
The aligned exchangeable model allows negative correlations within its
positive-semidefinite parameter space. The irregular exchangeable model has
nonnegative independent and random-intercept variance components.

The numerical implementation preserves the original initialization method,
update order, covariance optimizer settings, and stopping conditions.
Residuals and objective summaries are maintained incrementally. Covariance
geometry is reused, and subjects with identical irregular time-distance
matrices share covariance decompositions. Numerical results can differ at
floating-point precision from an R implementation.

Imputation retains the original objective evaluated after replacing missing
entries; this is not a separately derived observed-data objective.
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

Except for the greedy-only method, \code{elbo_steps} records the initial
objective and each noise, loading, score, and, when relevant, imputation update.
}
\examples{
set.seed(1)
x = array(rnorm(12 * 5 * 3), c(12, 5, 3))
fit = evblm(x, D = c(0, 1, 3), R = 1, max_iter = 2)
dim(fit$S)

times = list(c(0, 1), c(0, 2, 4), c(1, 3))
x_irregular = lapply(times, function(d) matrix(rnorm(12 * length(d)), 12))
fit_irregular = evblm(x_irregular, times, R = 1, max_iter = 2)
}
