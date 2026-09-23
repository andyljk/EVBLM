library(EVBLM)
set.seed(141)
p = 80
n = 30
D = 0:3
U = matrix(rnorm(p * 2), p, 2)
B = matrix(rnorm(n * 2), n, 2)
X = array(0, c(p, n, 4))
for (t in 1:4) {
  X[, , t] = U %*% t(sweep(B, 2, c(1 + .1*t, 1 - .1*t), '*')) + matrix(rnorm(p*n), p)
}

# At rank zero, the variance is exactly the observed second moment. A wholly
# missing visit borrows the observation-count-weighted pooled variance.
noise = array(rnorm(p*n*4), c(p,n,4))
noise[sample(length(noise), round(.3*length(noise)))] = NA
noise[, , 4] = NA
fit = evblm_impute(noise, D, R = 0)
counts = apply(!is.na(noise), 3, sum)
expected = apply(noise[, , 1:3]^2, 3, mean, na.rm = TRUE)
stopifnot(isTRUE(all.equal(1/fit$noise[1:3], expected, tolerance = 1e-6)),
  abs(1/fit$noise[4] - weighted.mean(expected, counts[1:3])) < 1e-6)

for (pattern in c('entrywise', 'blockwise', 'empty_visit')) {
  data = X
  if (pattern == 'entrywise') data[sample(length(data), length(data)/2)] = NA
  if (pattern == 'blockwise') for (t in 1:4) data[, sample(n,n/2), t] = NA
  if (pattern == 'empty_visit') data[, , 4] = NA
  for (prior in c('EC', 'RBF', 'Free')) {
    fit = evblm_impute(data, D, fn = prior, max_iter = 30, null_check = FALSE)
    stopifnot(fit$n_factors == 2, all(is.finite(fit$noise)),
      all(1/fit$noise > .75), all(1/fit$noise < 1.25))
    # Recompute the augmented bound directly from dense posterior moments.
    value = sum(fit$u$neg_KL) + sum(fit$v$neg_KL)
    for (t in 1:4) {
      mean = fit$u$pos[, , 1] %*% t(fit$v$pos[, , 1, t])
      variance = fit$u$pos[, , 2] %*% t(fit$v$pos[, , 2, t]) -
        (fit$u$pos[, , 1]^2) %*% t(fit$v$pos[, , 1, t]^2)
      observed = !is.na(data[, , t])
      rss = sum((data[, , t][observed] - mean[observed])^2) + sum(variance)
      value = value - .5 * (sum(observed)*log(2*pi/fit$noise[t]) + fit$noise[t]*rss)
    }
    stopifnot(abs(value - tail(fit$elbo,1)) < 1e-7*(1+abs(value)))
    e = fit$elbo_steps$elbo[is.finite(fit$elbo_steps$elbo)]
    stopifnot(all(diff(e) >= -1e-7*(1+abs(head(e,-1)))))
    if (pattern == 'empty_visit') {
      count = apply(!is.na(data), 3, sum)
      stopifnot(abs(1/fit$noise[4] - weighted.mean(1/fit$noise[1:3],count[1:3])) < 1e-8)
    }
  }
}

# Irregular blocks use one shared noise variance and the same augmented bound.
source(file.path('fixtures', 'inputs.R'))
for (prior in c('EC', 'RBF')) {
  fit = evblm_impute(irregular_missing, irregular_times, R = 2, fn = prior, max_iter = 20, null_check = FALSE)
  rss = 0
  count = 0
  for (i in seq_along(irregular_missing)) {
    mu = fit$u$pos[, , 1] %*% fit$v$pos[[i]][, , 1]
    variance = fit$u$pos[, , 2] %*% fit$v$pos[[i]][, , 2] -
      (fit$u$pos[, , 1]^2) %*% (fit$v$pos[[i]][, , 1]^2)
    observed = !is.na(irregular_missing[[i]])
    rss = rss + sum((irregular_missing[[i]][observed]-mu[observed])^2) + sum(variance)
    count = count + sum(observed)
  }
  value = -.5*(count*log(2*pi/fit$noise) + fit$noise*rss) + sum(fit$u$neg_KL) + sum(fit$v$neg_KL)
  stopifnot(abs(value-tail(fit$elbo,1)) < 1e-7*(1+abs(value)))
  e = fit$elbo_steps$elbo[is.finite(fit$elbo_steps$elbo)]
  stopifnot(all(diff(e) >= -1e-7*(1+abs(head(e,-1)))))
}
