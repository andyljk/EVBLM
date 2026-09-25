library(EVBLM)
source(file.path("fixtures", "inputs.R"))

default = evblm(aligned_data, aligned_times, R = 2, max_iter = 3)
explicit = evblm(aligned_data, aligned_times, R = 2, max_iter = 3, loading_prior = "normal")
stopifnot(identical(default, explicit), identical(default$u$prior, "normal"))
failure = tryCatch(evblm(aligned_data, aligned_times, R = 1,
  loading_prior = "horseshoe"), error = identity)
stopifnot(inherits(failure, "error"))

if (requireNamespace("ebnm", quietly = TRUE)) {
  families = c("point_normal", "point_laplace", "normal_scale_mixture")
  for (family in families) {
    for (layout in c("aligned", "irregular")) {
      X = get(paste0(layout, "_data"))
      D = get(paste0(layout, "_times"))
      covariances = if (layout == "aligned") c("EC", "Free", "RBF") else c("EC", "RBF")
      for (covariance in covariances) {
        for (impute in c(FALSE, TRUE)) {
          data = if (impute) get(paste0(layout, "_missing")) else X
          fit = if (impute) evblm_impute(data, D, R = 2, fn = covariance,
            loading_prior = family, max_iter = 3, thres = 0, null_check = FALSE) else
            evblm(data, D, R = 2, fn = covariance, loading_prior = family,
              max_iter = 3, thres = 0, null_check = FALSE)
          stopifnot(identical(fit$u$prior, family),
            all(is.finite(c(unlist(fit$S), fit$noise, fit$elbo))),
            all(fit$u$pos[, , 2] >= fit$u$pos[, , 1]^2 - 1e-10))
          trace = fit$elbo_steps$elbo[is.finite(fit$elbo_steps$elbo)]
          stopifnot(all(diff(trace) >= -1e-7 * (1 + abs(head(trace, -1)))))

          # Independently reconstruct the bound from dense posterior moments.
          U = matrix(fit$u$pos[, , 1], nrow = 6)
          U2 = matrix(fit$u$pos[, , 2], nrow = 6)
          if (layout == "aligned") {
            Y = matrix(aperm(data, c(1, 3, 2)), nrow = 6)
            V = matrix(aperm(fit$v$pos[, , 1, , drop = FALSE], c(4, 1, 2, 3)), ncol = 2)
            V2 = matrix(aperm(fit$v$pos[, , 2, , drop = FALSE], c(4, 1, 2, 3)), ncol = 2)
            tau = rep(fit$noise, dim(data)[2])
          } else {
            Y = do.call(cbind, data)
            V = do.call(rbind, lapply(fit$v$pos, function(v) t(v[, , 1])))
            V2 = do.call(rbind, lapply(fit$v$pos, function(v) t(v[, , 2])))
            tau = rep(fit$noise, ncol(Y))
          }
          count = colSums(!is.na(Y))
          signal = U %*% t(V)
          Y[is.na(Y)] = signal[is.na(Y)]
          rss = colSums((Y - signal)^2 + U2 %*% t(V2) - U^2 %*% t(V^2))
          bound = -.5 * sum(count * log(2 * pi / tau) + tau * rss) +
            sum(fit$u$neg_KL) + sum(fit$v$neg_KL)
          stopifnot(abs(bound - tail(fit$elbo, 1)) < 1e-8 * (1 + abs(bound)))
        }
      }

      # A one-sweep fit must match a direct EBNM call on the loading pseudo-data.
      Y = if (layout == "aligned") matrix(aperm(X, c(1, 3, 2)), nrow = 6) else do.call(cbind, X)
      v = seq(.5, 1.5, length.out = ncol(Y))
      u = list(pos = array(0, c(6, 1, 2)))
      scores = if (layout == "aligned") list(pos = array(0, c(5, 1, 2, 3))) else
        list(pos = lapply(D, function(d) array(0, c(1, length(d), 2))))
      if (layout == "aligned") {
        scores$pos[, 1, 1, ] = t(matrix(v, nrow = 3))
        scores$pos[, 1, 2, ] = t(matrix(v^2, nrow = 3))
      } else {
        offset = 0
        for (i in seq_along(D)) {
          index = offset + seq_along(D[[i]])
          scores$pos[[i]][1, , 1] = v[index]
          scores$pos[[i]][1, , 2] = v[index]^2
          offset = offset + length(index)
        }
      }
      fit = evblm(X, D, R = 1, u_mt = u, v_mt = scores, loading_prior = family,
        max_iter = if (layout == "aligned") 0 else 1, null_check = FALSE)
      tau = if (layout == "aligned") rep(fit$noise, 5) else rep(fit$noise, ncol(Y))
      precision = sum(tau * v^2)
      x = drop(Y %*% (tau * v)) / precision
      expected = ebnm::ebnm(x, 1 / sqrt(precision), prior_family = family, mode = 0,
        output = c("posterior_mean", "posterior_second_moment", "log_likelihood"))
      stopifnot(isTRUE(all.equal(drop(fit$u$pos[, 1, 1]), expected$posterior$mean, tolerance = 1e-10)),
        isTRUE(all.equal(drop(fit$u$pos[, 1, 2]), expected$posterior$second_moment, tolerance = 1e-10)),
        abs(fit$u$loglik - as.numeric(expected$log_likelihood)) < 1e-9)

      resumed = evblm(X, D, R = 1, u_mt = fit$u, v_mt = fit$v,
        loading_prior = family, max_iter = 1, null_check = FALSE)
      stopifnot(all(is.finite(resumed$elbo)))
      if (family == "normal_scale_mixture") {
        stopifnot(identical(resumed$u$par[[1]]$sd, fit$u$par[[1]]$sd))
      }
      failure = tryCatch(evblm(X, D, R = 1, u_mt = fit$u, v_mt = fit$v,
        loading_prior = "normal"), error = identity)
      stopifnot(inherits(failure, "error"), grepl("Initial loading prior", conditionMessage(failure)))

      # Zero score precision remains an absorbing null factor without passing Inf to ebnm.
      if (layout == "aligned") scores$pos[] = 0 else
        for (i in seq_along(scores$pos)) scores$pos[[i]][] = 0
      zero = evblm(X, D, R = 1, u_mt = u, v_mt = scores,
        loading_prior = family, max_iter = 1, null_check = FALSE)
      stopifnot(all(zero$u$pos == 0), is.null(zero$u$par[[1]]), zero$u$neg_KL == 0)
      for (method in c("single", "greedy", "greedy+backfit")) {
        invisible(capture.output(fit <- evblm(X, D, method = method,
          loading_prior = family, max_iter = 1)))
        stopifnot(identical(fit$u$prior, family), all(is.finite(unlist(fit$S))))
      }
    }
  }
}
