context("select_survivors")

points = matrix(
    c( # front 1
      # emoa puts always Inf weight on boundary points, so they always survive 
      # points 1 and points 4 have the highest hypervolume contributions 
      1, 4, 
      2, 2, 
      3.9, 1.1, 
      4, 1, 
      # front 2
      # points 5 and points 7 have the highest hypervolume contributions as boundary points
      2.2, 3.2, 
      4, 3,
      4.2, 1,
      # front 3
      6, 6
    ), byrow = FALSE, nrow = 2L
  )

points = as.data.frame(t(points))

test_that("nds_selection basics", {

  # Check whether n_select works correctly for all approaches
  for (tie_breaker in c("HV", "CD")) {
    n_surv = lapply(seq_row(points), function(n_select) length(select_survivors(points, n_select, ref_point = c(6, 6), minimize = c(TRUE, TRUE), method = "dominance_based", tie_breaker = tie_breaker)))
    expect_set_equal(unlist(n_surv), seq_row(points))
  }

  n_surv = lapply(seq_len(nrow(points) - 1), function(n_select) length(select_survivors(points, n_select, ref_point = c(6, 6), minimize = c(TRUE, TRUE), method = "indicator_based")))
  expect_set_equal(unlist(n_surv), seq_len(nrow(points) - 1))

  # randomize the entries in the dataset
  rndidx = sample(seq_row(points))

  # check that nondominated sorting works (if no tie-breaking is needed)
  ids = select_survivors(points[rndidx, ], 4, ref_point = c(6, 6), minimize = c(TRUE, TRUE), method = "dominance_based")
  expect_set_equal(rndidx[ids], 1:4)
  
  ids = select_survivors(points[rndidx, ], 7, ref_point = c(6, 6), minimize = c(TRUE, TRUE), method = "dominance_based")
  expect_set_equal(rndidx[ids], 1:7)

  # if the first front is broken according to CD
  # points 1 2 4 have highest crowding distance
  ids = select_survivors(points, n_select = 3, ref_point = c(6, 6), minimize = c(TRUE, TRUE), method = "dominance_based", tie_breaker = "CD")
  expect_set_equal(ids, c(1, 2, 4))

  # if we select one point according to CD, we should get one of the border points
  ids = select_survivors(points, n_select = 1, ref_point = c(6, 6), minimize = c(TRUE, TRUE), method = "dominance_based", tie_breaker = "CD")
  expect_subset(ids, c(1, 4))

  # If the tie_breaker is hypervolume, an we select 1 point, point c(2, 2) spans the highest volume
  ids = select_survivors(points, n_select = 1, ref_point = c(6, 6), minimize = c(TRUE, TRUE), method = "dominance_based", tie_breaker = "HV")
  expect_set_equal(ids, 2)

  # If tie_breaker is hypervolume, and we select 3 points, then points 1, 2, 4 span the highest hypervol
  ids = select_survivors(points, n_select = 3, ref_point = c(6, 6), minimize = c(TRUE, TRUE), method = "dominance_based", tie_breaker = "HV")
  expect_set_equal(ids, c(1, 2, 4))


  # TODO: Better checks for indicator-based
  # TODO: Check for minimization vs. maximization

})
