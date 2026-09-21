\name{mvEBNM}
\alias{mvEBNM}
\alias{irrEBNM}
\title{Fit a Gaussian Longitudinal Empirical Bayes Normal Means Model}
\description{
Estimate a longitudinal Gaussian prior covariance and condition its latent
scores on Gaussian pseudo-observations with known standard errors.
}
\usage{
mvEBNM(X, s, D, fn = c("EC", "Free", "RBF"), par_init = NULL)

irrEBNM(X, s, D, fn = c("EC", "RBF"), par_init = NULL)
}
\arguments{
  \item{X}{For \code{mvEBNM()}, a finite numeric subjects by visits matrix.
  For \code{irrEBNM()}, a list of finite numeric visit vectors, one per subject.}
  \item{s}{Positive known standard errors: one per aligned visit for
  \code{mvEBNM()}, or one common value for \code{irrEBNM()}.
  All-infinite standard errors define an uninformative block.}
  \item{D}{Aligned finite visit times, or a list of finite irregular time
  vectors matching the pseudo-observations. Repeated times are allowed.}
  \item{fn}{\code{"EC"}, \code{"RBF"}, or, for \code{mvEBNM()}, \code{"Free"}.}
  \item{par_init}{Optional previous \code{c(eta, rho)} or \code{c(eta, alpha)}
  parameters used as an additional optimizer start where applicable.
  \code{eta} is the finite nonnegative prior standard deviation, \code{rho}
  the exchangeable correlation, and \code{alpha} the nonnegative RBF length
  scale. Aligned EC requires finite \code{rho}; irregular EC permits
  \code{NA} for an unidentified correlation.}
}
\details{
The aligned exchangeable covariance permits negative correlations. The
irregular exchangeable covariance has nonnegative independent and
random-intercept variance components. RBF estimation includes the zero,
independence, and constant-process boundaries alongside finite-length starts.
Zero-length RBF covariance shares a latent value across repeated visit times.
Unidentified irregular correlation or length parameters are returned as
\code{NA}; an infinite length denotes a constant-process boundary where identified.
}
\value{
A list with \code{prior}, \code{par}, \code{pos}, \code{loglik}, and
\code{neg_KL}. Aligned \code{pos} has dimensions subjects by two by visits;
irregular \code{pos} contains one two by visits matrix per subject. The two
entries represent posterior mean and second moment. Additional fields report
the solver, convergence, optimization candidates, selected start, and
projected gradient when relevant to the covariance solver.
}
\examples{
set.seed(2)
x = matrix(rnorm(30), nrow = 10)
fit = mvEBNM(x, rep(1, 3), c(0, 1, 2), fn = "EC")
fit$par
}
