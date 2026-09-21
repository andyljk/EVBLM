# Fixed inputs use no RNG, so the fixtures also test deterministic fitting.
aligned_times = c(0, 0.8, 2.2)
aligned_data = array(0, c(6, 5, 3))
for (visit in seq_along(aligned_times)) {
  aligned_data[, , visit] = outer(seq(-1.5, 1.5, length.out = 6),
    seq(0.7, 1.6, length.out = 5) * (1 + 0.3 * aligned_times[visit])) +
    outer(c(1, -1, 0.5, -0.5, 1.2, -1.2),
      cos(seq_len(5) + aligned_times[visit])) +
    matrix(0.2 * sin(seq_len(30) * 1.7 + visit), 6, 5)
}
aligned_missing = aligned_data
aligned_missing[c(2, 18, 48)] = NA

irregular_times = list(c(0, 0.8, 2.2), c(0, 1.3), c(0.4, 1, 2.5), c(0, 1.7))
irregular_data = lapply(seq_along(irregular_times), function(subject) {
  times = irregular_times[[subject]]
  outer(seq(-1.5, 1.5, length.out = 6),
    (0.5 + 0.3 * subject) * (1 + 0.3 * times)) +
    outer(c(1, -1, 0.5, -0.5, 1.2, -1.2), cos(subject + times)) +
    matrix(0.2 * sin(seq_len(6 * length(times)) * 1.7 + subject), 6)
})
irregular_missing = irregular_data
irregular_missing[[1]][2] = NA
irregular_missing[[3]][11] = NA
