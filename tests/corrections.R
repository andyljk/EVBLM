library(EVBLM)
source(file.path("fixtures", "inputs.R"))

# Exact SVD fits must acquire posterior uncertainty before estimating noise.
# The saturated case is ordinary noisy data fitted at full matrix rank.
set.seed(131)
cases = list(
  constant = array(1, c(6, 4, 3)),
  rank_one = array(outer(rnorm(6), rnorm(12)), c(6, 4, 3)),
  saturated = array(rnorm(72), c(6, 4, 3))
)
for (name in names(cases)) {
  for (prior in c("EC", "Free", "RBF")) {
    fit = evblm(cases[[name]], 0:2, R = if (name == "saturated") 6 else 1,
      fn = prior, max_iter = 10, thres = 0, null_check = FALSE)
    stopifnot(all(is.finite(c(fit$S, fit$noise, fit$elbo))), all(fit$noise > 0))
    finite_elbo = fit$elbo_steps$elbo[is.finite(fit$elbo_steps$elbo)]
    stopifnot(all(diff(finite_elbo) >= -1e-7 * (1 + abs(head(finite_elbo, -1)))))
  }
}

# Noise initialization must respect measurement units, including when RSS is
# numerically zero. Test both exact fits and ordinary noisy observations.
for (X in list(cases$rank_one, aligned_data)) {
  times = if (identical(dim(X), dim(aligned_data))) aligned_times else 0:2
  rank = if (identical(dim(X), dim(aligned_data))) 2 else 1
  for (prior in c("EC", "Free", "RBF")) {
    base = evblm(X, times, R = rank, fn = prior, max_iter = 8, thres = 0, null_check = FALSE)
    for (scale in c(0.01, 100)) {
      fit = evblm(X * scale, times, R = rank, fn = prior,
        max_iter = 8, thres = 0, null_check = FALSE)
      stopifnot(isTRUE(all.equal(fit$S / scale, base$S, tolerance = 2e-5)),
        isTRUE(all.equal(fit$noise * scale^2, base$noise, tolerance = 2e-5)))
    }
  }
}

# A failed decomposition must report its cause, rather than use empty outputs.
failure = tryCatch(mvEBNM(matrix(1e200, 3, 2), c(1, 1), 0:1, "Free"), error = identity)
stopifnot(inherits(failure, "error"),
  grepl("Aligned Free covariance eigendecomposition failed", conditionMessage(failure), fixed = TRUE))

# Locate the first crossing of Algorithm 4's missing-entry criterion using
# independently truncated runs. Observed entries must not enter that criterion.
missing = which(is.na(unlist(irregular_missing)))
threshold = 1e-5
for (prior in c("EC", "RBF")) {
  fit = evblm_impute(irregular_missing, irregular_times, R = 2, fn = prior,
    thres = threshold, max_iter = 50, null_check = FALSE)
  iteration = max(fit$elbo_steps$iteration)
  stopifnot(iteration >= 2, iteration < 50)
  previous = evblm_impute(irregular_missing, irregular_times, R = 2, fn = prior,
    thres = 0, max_iter = iteration - 1, null_check = FALSE)
  earlier = evblm_impute(irregular_missing, irregular_times, R = 2, fn = prior,
    thres = 0, max_iter = iteration - 2, null_check = FALSE)
  change = sum((unlist(fit$S)[missing] - unlist(previous$S)[missing])^2)
  previous_change = sum((unlist(previous$S)[missing] - unlist(earlier$S)[missing])^2)
  stopifnot(change <= threshold, previous_change > threshold)
}

# A small loading must be pruned even when the corresponding scores are not
# small. Rescale a valid warm start to isolate the loading-only criterion.
initial = evblm(aligned_data, aligned_times, R = 1, fn = "EC", null_check = FALSE)
u = initial$u
scale = 1e-10
u$pos[, , 1] = u$pos[, , 1] * scale
u$pos[, , 2] = u$pos[, , 2] * scale^2
u$par = lapply(u$par, function(par) par * scale)
u$loglik = u$loglik - dim(aligned_data)[1] * log(scale)
retained = evblm(aligned_data * scale, aligned_times, R = 1, fn = "EC",
  u_mt = u, v_mt = initial$v, max_iter = 0, null_check = FALSE)
stopifnot(sum(abs(retained$u$pos[, , 1])) < 1e-8,
  all(colSums(abs(matrix(retained$v$pos[, , 1, ], nrow = dim(aligned_data)[2]))) >= 1e-8))
pruned = evblm(aligned_data * scale, aligned_times, R = 1, fn = "EC",
  u_mt = u, v_mt = initial$v, max_iter = 0, null_check = TRUE)
stopifnot(pruned$n_factors == 0, dim(pruned$u$pos)[2] == 0,
  dim(pruned$v$pos)[2] == 0, length(pruned$u$par) == 0, length(pruned$v$par) == 0)
