library(EVBLM)
source(file.path("fixtures", "inputs.R"))
source(file.path("fixtures", "reference.R"))

# Optimizer starts may tie after floating-point reordering. Compare the fitted
# quantities and posterior second moments, which do not depend on SVD signs.
# 2e-5 allows accumulated numerical differences across coordinate updates.
set.seed(921)
initial_rng_state = .Random.seed
for (variant in c("aligned", "irregular")) {
  times = get(paste0(variant, "_times"))
  priors = if (variant == "aligned") c("EC", "Free", "RBF") else c("EC", "RBF")
  for (prior in priors) {
    for (impute in c(FALSE, TRUE)) {
      X = get(paste0(variant, if (impute) "_missing" else "_data"))
      fit = evblm(X, times, R = 2, fn = prior, method = "backfit",
        impute = impute, thres = 0, max_iter = 3)
      name = paste(variant, prior, if (impute) "impute" else "backfit", sep = "_")
      expected = reference[[name]]
      actual = list(S = fit$S, noise = fit$noise, elbo = fit$elbo,
        u_second = fit$u$pos[, , 2, drop = FALSE],
        v_second = if (variant == "aligned") fit$v$pos[, , 2, , drop = FALSE] else
          lapply(fit$v$pos, function(v) v[, , 2, drop = FALSE]),
        elbo_steps = fit$elbo_steps)
      for (field in names(expected)) {
        comparison = all.equal(actual[[field]], expected[[field]], tolerance = 2e-5)
        if (!isTRUE(comparison)) {
          stop(name, " / ", field, ": ", paste(comparison, collapse = "; "))
        }
      }
      stopifnot(identical(.Random.seed, initial_rng_state))
    }
  }

  # Aligned backfitting preserves its inclusive limit and runs one sweep at
  # max_iter = 0; irregular backfitting runs zero. Check both original behaviors.
  fit = evblm(get(paste0(variant, "_data")), times, R = 2, fn = "EC",
    method = "backfit", thres = 0, max_iter = 0, null_check = FALSE)
  expected = reference[[paste0(variant, "_zero_iterations")]]
  for (field in names(expected)) {
    comparison = all.equal(fit[[field]], expected[[field]], tolerance = 2e-5)
    if (!isTRUE(comparison)) {
      stop(variant, " max_iter = 0 / ", field, ": ", paste(comparison, collapse = "; "))
    }
  }
}

# The contrast-only EC boundary permits rho = -1 with two aligned visits.
X = cbind(c(-3, 3), c(3, -3))
s = 0.5
fit = mvEBNM(X, rep(s, 2), 0:1, fn = "EC")
contrast_variance = 18 - s^2
shrink = contrast_variance / 18
expected_mean = shrink * X
expected_second = expected_mean^2 + s^2 * shrink / 2
stopifnot(isTRUE(all.equal(unname(fit$par), c(sqrt(contrast_variance / 2), -1), tolerance = 1e-10)),
  isTRUE(all.equal(fit$pos[, 1, ], expected_mean, tolerance = 1e-10)),
  isTRUE(all.equal(fit$pos[, 2, ], expected_second, tolerance = 1e-10)))

# A constant RBF schedule gives a rank-one prior and an infinite length scale.
X = matrix(c(1, 2, 3, 1.5, 2.5, 3.5), 3, 2)
s = 0.5
variance = max(mean(rowMeans(X)^2) - s^2 / ncol(X), 0)
shrink = ncol(X) * variance / (s^2 + ncol(X) * variance)
expected_mean = matrix(shrink * rowMeans(X), nrow(X), ncol(X))
expected_second = expected_mean^2 + s^2 * shrink / ncol(X)
fit = mvEBNM(X, rep(s, 2), c(0, 0), fn = "RBF")
stopifnot(is.infinite(fit$par["alpha"]),
  isTRUE(all.equal(unname(fit$par["eta"]), sqrt(variance), tolerance = 1e-10)),
  isTRUE(all.equal(fit$pos[, 1, ], expected_mean, tolerance = 1e-10)),
  isTRUE(all.equal(fit$pos[, 2, ], expected_second, tolerance = 1e-10)))

for (prior in c("EC", "Free", "RBF")) {
  fit = mvEBNM(X, rep(Inf, 2), 0:1, fn = prior)
  stopifnot(all(fit$pos == 0), fit$loglik == 0, fit$neg_KL == 0)
  fit = mvEBNM(matrix(0, 3, 2), rep(s, 2), 0:1, fn = prior)
  stopifnot(all(fit$pos == 0), fit$neg_KL == 0)
}
for (prior in c("EC", "RBF")) {
  fit = irrEBNM(list(c(1, 2), c(2, 3)), Inf, list(0:1, 0:1), fn = prior)
  stopifnot(all(unlist(fit$pos) == 0), fit$loglik == 0, fit$neg_KL == 0)
  fit = irrEBNM(list(c(0, 0), c(0, 0)), s, list(0:1, 0:1), fn = prior)
  stopifnot(all(unlist(fit$pos) == 0), fit$neg_KL == 0)
}
stopifnot(identical(.Random.seed, initial_rng_state))

# Invalid public inputs must fail validation before entering the native engine.
invalid_calls = list(
  quote(evblm(aligned_data, aligned_times[-1], R = 2)),
  quote(evblm(aligned_missing, aligned_times, R = 2)),
  quote(evblm(irregular_data, irregular_times, R = 2, fn = "Free")),
  quote(evblm(aligned_data, aligned_times, R = -1)),
  quote(mvEBNM(matrix(1, 3, 2), c(1, 1), 0:1, fn = "EC", par_init = c(1, NA)))
)
for (call in invalid_calls) {
  stopifnot(inherits(tryCatch(eval(call), error = identity), "error"))
}
