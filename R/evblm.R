# The R interface validates data and converts between the public tensor/list
# shapes and the subject-major matrices used by the compiled fitting engine.
evblm = function(X, D, R = NULL, fn = "EC",
                 method = if (is.null(R)) "greedy+backfit" else "backfit",
                 impute = FALSE, thres = 1e-8, max_iter = 50,
                 verbose = FALSE, null_check = TRUE,
                 u_mt = NULL, v_mt = NULL, D_init = NULL) {
  aligned = is.array(X) && length(dim(X)) == 3L
  if (aligned) {
    if (!is.numeric(X) || any(dim(X) == 0L)) {
      stop("Aligned X must be a nonempty numeric variables x subjects x visits array.")
    }
    if (!is.numeric(D) || length(D) != dim(X)[3L] || any(!is.finite(D))) {
      stop("D must contain one finite time per aligned visit.")
    }
    variables = dim(X)[1L]
    subjects = dim(X)[2L]
    visits = length(D)
    counts = rep.int(visits, subjects)
    schedules = list(as.numeric(D))
    packed = matrix(aperm(X, c(1L, 3L, 2L)), nrow = variables)
    fn = match.arg(fn, c("EC", "Free", "RBF"))
    if (!is.null(D_init)) stop("D_init is only used for irregular input.")
  } else {
    if (!is.list(X) || !length(X) ||
        !all(vapply(X, function(x) is.matrix(x) && is.numeric(x) && all(dim(x) > 0L), logical(1)))) {
      stop("Irregular X must be a nonempty list of numeric variables x visits matrices.")
    }
    variables = nrow(X[[1L]])
    subjects = length(X)
    if (any(vapply(X, nrow, integer(1)) != variables)) {
      stop("Every subject matrix must contain the same number of variables.")
    }
    counts = vapply(X, ncol, integer(1))
    if (!is.list(D) || length(D) != subjects ||
        !all(vapply(D, function(d) is.numeric(d) && all(is.finite(d)), logical(1))) ||
        any(lengths(D) != counts)) {
      stop("D must contain a finite time vector matching each subject's visits.")
    }
    if (!is.null(D_init) && (!is.list(D_init) || length(D_init) != length(D) ||
                            any(lengths(D_init) != lengths(D)))) {
      stop("D_init must have the same subject and visit counts as D.")
    }
    schedules = lapply(D, as.numeric)
    packed = do.call(cbind, X)
    fn = match.arg(fn, c("EC", "RBF"))
  }
  method = match.arg(method, c("single", "backfit", "greedy", "greedy+backfit"))
  if (!is.logical(impute) || length(impute) != 1L || is.na(impute) ||
      !is.logical(verbose) || length(verbose) != 1L || is.na(verbose) ||
      !is.logical(null_check) || length(null_check) != 1L || is.na(null_check)) {
    stop("impute, verbose, and null_check must each be TRUE or FALSE.")
  }
  if (any(is.infinite(packed)) || (!impute && anyNA(packed))) {
    stop("X must be finite; missing values are allowed only when impute = TRUE.")
  }
  if (all(is.na(packed) | packed == 0)) {
    stop("X must contain a nonzero observed entry to initialize noise precision.")
  }
  if (impute && !method %in% c("backfit", "greedy+backfit")) {
    stop("Imputation requires method = 'backfit' or 'greedy+backfit'.")
  }
  if (!is.numeric(thres) || length(thres) != 1L || !is.finite(thres) || thres < 0 ||
      !is.numeric(max_iter) || length(max_iter) != 1L || !is.finite(max_iter) ||
      max_iter < 0 || max_iter > .Machine$integer.max || max_iter != floor(max_iter)) {
    stop("thres must be nonnegative and finite; max_iter must be a nonnegative integer.")
  }
  if (method == "single") R = 1L
  if (method == "backfit" && is.null(R)) stop("R is required for backfitting.")
  if (!is.null(R) && (!is.numeric(R) || length(R) != 1L || !is.finite(R) ||
                     R < 0 || R > min(dim(packed)) || R != floor(R))) {
    stop("R must be a nonnegative integer no larger than min(n_variables, total_visits).")
  }
  total_visits = sum(counts)
  initial_u = initial_v = NULL
  if (xor(is.null(u_mt), is.null(v_mt))) stop("Supply both u_mt and v_mt, or neither.")
  if (!is.null(u_mt)) {
    if (method != "backfit" || impute) stop("Initial factors are supported for non-imputing backfitting.")
    if (!is.list(u_mt) || !is.list(v_mt) ||
        !identical(dim(u_mt$pos), c(variables, as.integer(R), 2L))) {
      stop("u_mt$pos must have dimensions variables x R x 2.")
    }
    initial_u = list(
      mean = matrix(u_mt$pos[, , 1L], variables, R),
      second = matrix(u_mt$pos[, , 2L], variables, R),
      par = if (!length(u_mt$par)) vector("list", R) else u_mt$par,
      loglik = as.numeric(u_mt$loglik), neg_KL = as.numeric(u_mt$neg_KL)
    )
    if (aligned) {
      if (!identical(dim(v_mt$pos), c(subjects, as.integer(R), 2L, visits))) {
        stop("Aligned v_mt$pos must have dimensions subjects x R x 2 x visits.")
      }
      v_mean = matrix(aperm(v_mt$pos[, , 1L, , drop = FALSE], c(4L, 1L, 2L, 3L)), total_visits, R)
      v_second = matrix(aperm(v_mt$pos[, , 2L, , drop = FALSE], c(4L, 1L, 2L, 3L)), total_visits, R)
    } else {
      if (!is.list(v_mt$pos) || length(v_mt$pos) != subjects ||
          !all(vapply(seq_len(subjects), function(i) {
            identical(dim(v_mt$pos[[i]]), c(as.integer(R), counts[i], 2L))
          }, logical(1)))) {
        stop("Irregular v_mt$pos must contain one R x visits x 2 array per subject.")
      }
      v_mean = do.call(rbind, lapply(v_mt$pos, function(v) t(matrix(v[, , 1L], R, dim(v)[2L]))))
      v_second = do.call(rbind, lapply(v_mt$pos, function(v) t(matrix(v[, , 2L], R, dim(v)[2L]))))
    }
    initial_v = list(mean = v_mean, second = v_second,
      par = if (!length(v_mt$par)) vector("list", R) else v_mt$par,
      loglik = as.numeric(v_mt$loglik), neg_KL = as.numeric(v_mt$neg_KL))
    for (initial in list(initial_u, initial_v)) {
      if (any(!is.finite(initial$mean)) || any(!is.finite(initial$second)) ||
          !is.list(initial$par) || length(initial$par) != R ||
          !length(initial$loglik) %in% c(0L, R) || !length(initial$neg_KL) %in% c(0L, R) ||
          length(initial$loglik) != length(initial$neg_KL)) {
        stop("Initial factors must have finite moments and one parameter entry per factor.")
      }
    }
    if (length(initial_u$neg_KL) != length(initial_v$neg_KL)) {
      stop("Loading and score initializations must both provide complete likelihood and KL metadata, or neither.")
    }
    for (par in initial_u$par) {
      if (!is.null(par) && (!is.numeric(par) || length(par) != 2L ||
                            any(!is.finite(par)) || par[2L] < 0)) {
        stop("Each loading parameter must contain a finite mean and nonnegative finite standard deviation.")
      }
    }
    for (par in initial_v$par) {
      if (is.null(par)) next
      if (fn == "Free") {
        if (!is.matrix(par) || !is.numeric(par) ||
            !identical(dim(par), c(visits, visits)) || any(!is.finite(par))) {
          stop("Free covariance parameters must be finite visits x visits matrices.")
        }
      } else {
        if (!is.numeric(par) || length(par) != 2L || !is.finite(par[1L]) || par[1L] < 0) {
          stop("Each score parameter must contain eta and one covariance parameter, with finite nonnegative eta.")
        }
        if (aligned && fn == "EC" && is.na(par[2L])) {
          stop("Aligned EC initial correlation must be finite; use zero for a zero-variance prior.")
        }
        if (!is.na(par[2L]) && ((fn == "RBF" && par[2L] < 0) ||
            (fn == "EC" && (!is.finite(par[2L]) || par[2L] > 1 ||
                             par[2L] < if (aligned && visits > 1L) -1 / (visits - 1L) else 0)))) {
          stop("Initial score covariance parameters are outside the model's parameter range.")
        }
      }
    }
    if (xor(length(v_mt$optimization) > 0L, length(v_mt$converged) > 0L)) {
      stop("Supply both v_mt$optimization and v_mt$converged, or neither.")
    }
    if (length(v_mt$optimization)) {
      if (!is.list(v_mt$optimization) || length(v_mt$optimization) != R) {
        stop("v_mt$optimization must contain one entry per factor.")
      }
      initial_v$optimization = v_mt$optimization
    }
    if (length(v_mt$converged)) {
      if (!is.logical(v_mt$converged) || length(v_mt$converged) != R) {
        stop("v_mt$converged must contain one logical value per factor.")
      }
      initial_v$converged = v_mt$converged
    }
  }
  fit = evblm_engine_cpp(packed, schedules, aligned, subjects, fn, method,
    if (is.null(R)) 0L else as.integer(R), impute, thres, as.integer(max_iter),
    verbose, null_check, initial_u, initial_v)

  rank = ncol(fit$u$mean)
  fit$u$pos = array(c(fit$u$mean, fit$u$second), c(variables, rank, 2L))
  fit$u$mean = fit$u$second = NULL
  if (aligned) {
    pos = array(0, c(subjects, rank, 2L, visits))
    pos[, , 1L, ] = aperm(array(fit$v$mean, c(visits, subjects, rank)), c(2L, 3L, 1L))
    pos[, , 2L, ] = aperm(array(fit$v$second, c(visits, subjects, rank)), c(2L, 3L, 1L))
    fit$v$pos = pos
    fit$S = aperm(array(fit$S, c(variables, visits, subjects)), c(1L, 3L, 2L))
  } else {
    fit$v$pos = fit$S_split = vector("list", subjects)
    offset = 0L
    for (i in seq_len(subjects)) {
      columns = offset + seq_len(counts[i])
      fit$v$pos[[i]] = array(c(t(fit$v$mean[columns, , drop = FALSE]),
        t(fit$v$second[columns, , drop = FALSE])), c(rank, counts[i], 2L))
      fit$S_split[[i]] = fit$S[, columns, drop = FALSE]
      offset = offset + counts[i]
    }
    fit$S = fit$S_split
    fit$S_split = NULL
  }
  fit$v$mean = fit$v$second = NULL
  if (method == "single") fit$n_factors = NULL else fit$n_iter = NULL
  if (method == "greedy") fit$elbo_steps = NULL
  fit
}

evblm_impute = function(X, D, R = NULL, fn = "EC",
                        method = if (is.null(R)) "greedy+backfit" else "backfit",
                        thres = 1e-5, max_iter = 100, verbose = FALSE,
                        null_check = TRUE, D_init = NULL) {
  evblm(X, D, R = R, fn = fn, method = method, impute = TRUE,
    thres = thres, max_iter = max_iter, verbose = verbose,
    null_check = null_check, D_init = D_init)
}

mvEBNM = function(X, s, D, fn = c("EC", "Free", "RBF"), par_init = NULL) {
  fn = match.arg(fn)
  if (!is.matrix(X) || !is.numeric(X) || any(dim(X) == 0L) || any(!is.finite(X))) {
    stop("X must be a nonempty finite numeric subjects x visits matrix.")
  }
  if (!is.numeric(D) || length(D) != ncol(X) || any(!is.finite(D)) ||
      !is.numeric(s) || length(s) != ncol(X) || anyNA(s) || any(s <= 0) ||
      (any(is.infinite(s)) && !all(is.infinite(s)))) {
    stop("D and s must match the visits; times must be finite and standard errors positive, with either all finite or all infinite.")
  }
  if (!is.null(par_init) && fn != "Free" &&
      (!is.numeric(par_init) || length(par_init) != 2L || !is.finite(par_init[1L]) || par_init[1L] < 0)) {
    stop("par_init must contain eta and the covariance parameter, with finite nonnegative eta.")
  }
  if (!is.null(par_init) && fn == "EC" &&
      (!is.finite(par_init[2L]) || par_init[2L] > 1 ||
       par_init[2L] < if (ncol(X) > 1L) -1 / (ncol(X) - 1L) else 0)) {
    stop("Aligned EC initial correlation must be finite and within its covariance parameter range.")
  }
  if (!is.null(par_init) && fn == "RBF" && !is.na(par_init[2L]) && par_init[2L] < 0) {
    stop("The initial RBF length must be nonnegative, NA, or infinite.")
  }
  fit = mv_ebnm_cpp(X, s, D, fn, par_init)
  fit$pos = array(0, c(nrow(X), 2L, ncol(X)))
  fit$pos[, 1L, ] = fit$mean
  fit$pos[, 2L, ] = fit$second
  fit$mean = fit$second = NULL
  fit
}

irrEBNM = function(X, s, D, fn = c("EC", "RBF"), par_init = NULL) {
  fn = match.arg(fn)
  if (!is.list(X) || !length(X) || !is.list(D) || length(D) != length(X) ||
      !all(vapply(X, function(x) is.numeric(x) && length(x) > 0L && all(is.finite(x)), logical(1))) ||
      !all(vapply(D, function(d) is.numeric(d) && all(is.finite(d)), logical(1))) ||
      any(lengths(X) != lengths(D))) {
    stop("X and D must be matching lists of nonempty finite numeric observations and times.")
  }
  if (!is.numeric(s) || length(s) != 1L || is.na(s) || s <= 0) {
    stop("s must be one positive standard error, possibly infinite.")
  }
  if (!is.null(par_init) && (!is.numeric(par_init) || length(par_init) != 2L ||
                           !is.finite(par_init[1L]) || par_init[1L] < 0)) {
    stop("par_init must contain eta and the covariance parameter, with finite nonnegative eta.")
  }
  if (!is.null(par_init) && !is.na(par_init[2L]) &&
      ((fn == "EC" && (!is.finite(par_init[2L]) || par_init[2L] < 0 || par_init[2L] > 1)) ||
       (fn == "RBF" && par_init[2L] < 0))) {
    stop("The initial irregular covariance parameter is outside the model's parameter range.")
  }
  fit = irr_ebnm_cpp(lapply(X, as.numeric), s, lapply(D, as.numeric), fn, par_init)
  fit$pos = vector("list", length(X))
  offset = 0L
  for (i in seq_along(X)) {
    columns = offset + seq_along(X[[i]])
    mu = as.numeric(fit$mean[columns])
    fit$pos[[i]] = rbind(mu, as.numeric(fit$second[columns]))
    if (is.infinite(s)) dimnames(fit$pos[[i]]) = NULL
    offset = offset + length(X[[i]])
  }
  fit$mean = fit$second = NULL
  fit
}
