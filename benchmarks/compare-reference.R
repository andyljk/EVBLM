library(EVBLM)
reference_dir = "/Users/andyl/Desktop/Bayesian_BIDIFAC/CSDA_Submission"
aligned = new.env()
irregular = new.env()
suppressPackageStartupMessages(sys.source(
  file.path(reference_dir, "EVBLM_functions.R"), aligned))
suppressPackageStartupMessages(sys.source(
  file.path(reference_dir, "EVBLM_irreg.R"), irregular))
set.seed(2026)
variables = 120
subjects = 24
visits = 5
rank = 3
U = matrix(rnorm(variables * rank), variables)
coefficients = matrix(rnorm(subjects * rank), subjects)
D = c(0, 1, 2, 4, 6)
X = array(0, c(variables, subjects, visits))
for (m in seq_len(visits)) {
  scores = coefficients * rep(c(1 + 0.2 * D[m], sin(D[m] / 2) + 0.4, cos(D[m] / 3)), each = subjects)
  X[, , m] = U %*% t(scores) + matrix(rnorm(variables * subjects), variables)
}
irregular_D = lapply(seq_len(subjects), function(i) {
  if (i %% 2) D else sort(runif(4 + i %% 3, 0, 6))
})
irregular_X = lapply(seq_len(subjects), function(i) {
  t = irregular_D[[i]]
  scores = cbind(coefficients[i, 1] * (1 + 0.2 * t),
    coefficients[i, 2] * (sin(t / 2) + 0.4), coefficients[i, 3] * cos(t / 3))
  U %*% t(scores) + matrix(rnorm(variables * length(t)), variables)
})
for (variant in c("aligned", "irregular")) {
  data = if (variant == "aligned") X else irregular_X
  times = if (variant == "aligned") D else irregular_D
  reference = if (variant == "aligned") aligned else irregular
  for (prior in if (variant == "aligned") c("EC", "Free", "RBF") else c("EC", "RBF")) {
    elapsed_R = elapsed_cpp = numeric(3)
    for (repeat_index in 1:3) {
      gc()
      elapsed_R[repeat_index] = system.time(
        original <- reference$backfit(data, times, R = rank, fn = prior,
          max_iter = 4, thres = 0, null_check = FALSE)
      )[["elapsed"]]
      gc()
      elapsed_cpp[repeat_index] = system.time(
        compiled <- EVBLM::evblm(data, times, R = rank, fn = prior,
          max_iter = 4, thres = 0, null_check = FALSE)
      )[["elapsed"]]
    }
    signal_relative = sqrt(sum((unlist(original$S) - unlist(compiled$S))^2) / sum(unlist(original$S)^2))
    cat(variant, prior, "R", median(elapsed_R), "C++", median(elapsed_cpp),
      "ratio", median(elapsed_R) / median(elapsed_cpp),
      "signal relative error", format(signal_relative, digits = 3), "\n")
  }
}
