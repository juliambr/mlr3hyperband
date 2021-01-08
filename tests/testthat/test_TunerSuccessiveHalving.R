library(mlr3learners)
library(mlr3pipelines)
library(checkmate)

context("TunerSuccessiveHalving")

test_that("TunerSuccessiveHalving singlecrit", {

	par.set = ParamSet$new(params = list(
		ParamInt$new("nrounds", lower = 2, upper = 2^3,
		  tags = "budget"),
		ParamInt$new("max_depth", lower = 1, upper = 100)
	))

	task = tsk("pima")

	learner = lrn("classif.xgboost")

	term = trm("none")

    inst = TuningInstanceSingleCrit$new(task = task, learner = learner,
      resampling =  rsmp("holdout"), measure = msr("classif.acc"), terminator = term,
      search_space = par.set)

    # If we do not specify r, then it is computed from the parameter set 
    tuner = mlr3tuning::tnr("sh", eta = 2, n = 24)
    # given lower = 2 and upper = 2^4 we have 4 stages 
    tuner$optimize(inst)
    archive = inst$archive$data
    # total budget should be 
    expect_equal(sum(archive$nrounds), 3 * as.numeric(tuner$param_set$values$n * par.set$lower["nrounds"]))
    expect_equal(as.numeric(table(archive$nrounds)), 24 / 2^(0:2))

    # make sure that the best value survives 
    expect_true(archive[stage == 1, ][, .SD[which.max(classif.acc)], ]$max_depth %in% archive[stage == 2, ]$max_depth)

    # If we r is higher then the minimum budget, the number of successive halvings should reduce
    inst = TuningInstanceSingleCrit$new(task = task, learner = learner,
      resampling =  rsmp("holdout"), measure = msr("classif.acc"), terminator = term,
      search_space = par.set)

    tuner = mlr3tuning::tnr("sh", eta = 2, n = 24, r = 4)
    tuner$optimize(inst)
    archive = inst$archive$data
    # total budget should be 
    expect_equal(sum(archive$nrounds), 2 * as.numeric(tuner$param_set$values$n * 4))

    tuner = mlr3tuning::tnr("sh", eta = 2, n = 24, r = 2^4 + 1)
    expect_error(tuner$optimize(inst))
})


test_that("TunerSuccessiveHalving multicrit", {

	par.set = ParamSet$new(params = list(
		ParamInt$new("nrounds", lower = 2, upper = 2^3,
		  tags = "budget"),
		ParamInt$new("max_depth", lower = 1, upper = 100)
	))

	task = tsk("pima")

	learner = lrn("classif.xgboost")

	term = trm("none")

    measures = c("classif.tpr", "classif.fpr")

    inst = TuningInstanceMultiCrit$new(task = task, learner = learner,
      resampling =  rsmp("holdout"), measure = msrs(measures), terminator = term,
      search_space = par.set)

    # check if the indicator-based method works
    tuner = mlr3tuning::tnr("sh", eta = 2, n = 24, mo_method = "indicator_based", np = 3)

    # expect a total budget of np * nstages = n * r = 3 * 3 * 24 * 2
    tuner$optimize(inst)
    archive = inst$archive$data
    expect_equal(sum(archive$nrounds), 3 * 3 * 24 * 2)

    inst = TuningInstanceMultiCrit$new(task = task, learner = learner,
      resampling =  rsmp("holdout"), measure = msrs(measures), terminator = term,
      search_space = par.set)

    # check if the indicator-based method works
    tuner = mlr3tuning::tnr("sh", eta = 2, n = 24, mo_method = "dominance_based", np = 8, tie_breaker = "CD")

    # We only can do two iterations instead of 3 if we want to reduce down to 8
    tuner$optimize(inst)
    archive = inst$archive$data
    expect_equal(sum(archive$nrounds), 2 * 24 * 2)

    inst = TuningInstanceMultiCrit$new(task = task, learner = learner,
      resampling =  rsmp("holdout"), measure = msrs(measures), terminator = term,
      search_space = par.set)

    tuner = mlr3tuning::tnr("sh", eta = 2, n = 24, mo_method = "dominance_based", np = 8, tie_breaker = "HV")
    expect_success(tuner$optimize(inst))
})


test_that("TunerSuccessiveHalving singlecrit", {

	par.set = ParamSet$new(params = list(
		ParamInt$new("nrounds", lower = 2, upper = 2^5,
		  tags = "budget"),
		ParamInt$new("max_depth", lower = 1, upper = 100)
	))

	task = tsk("pima")

	learner = lrn("classif.xgboost")

	term = trm("none")

    measures = c("classif.tpr", "classif.fpr")

    inst = TuningInstanceMultiCrit$new(task = task, learner = learner,
      resampling =  rsmp("holdout"), measure = msrs(measures), terminator = term,
      search_space = par.set)



    tuner = mlr3tuning::tnr("sh", eta = 2, n = 2^5, r = 2^6, mo_method = "indicator_based", np = 2, tie_breaker = "CD")

    tuner$optimize(inst)


    inst = TuningInstanceMultiCrit$new(task = task, learner = learner,
      resampling =  rsmp("holdout"), measure = msrs(measures), terminator = term,
      search_space = par.set)

    tuner = mlr3tuning::tnr("sh", eta = 2, n = 2^5, r = 20, mo_method = "dominance_based", np = 1, tie_breaker = "HV")

    tuner$optimize(inst)
})



test_that("TunerSuccessiveHalving synthetic", {

	mlr_tuners$add("sh", TunerSuccessiveHalving)	

	# Define objective function
	fun = function(xs) {
		# This is a simple multi-objective function
		# We sample a noise within a specific radius
		# The higher the budget, the smaller the noise
		radius = 1 / xs[[3]]
		phi = runif(1, 0, 2 * pi)
		noise = c(radius * cos(phi), radius * sin(phi)) 
		fun = c(xs[[1]]^2, - xs[[1]] + 3 + xs[[2]])
		res = fun + noise
	}

	# Set domain
	domain = ParamSet$new(list(
		ParamDbl$new("x1", 0, 3),
		ParamDbl$new("x2", 0, 4),
		ParamDbl$new("x3", 2, 8, tag = "budget")
	))

	# Set codomain
	codomain = ParamSet$new(list(
	  ParamDbl$new("y1", tags = "minimize"),
	  ParamDbl$new("y2", tags = "minimize")
	))

	obj = ObjectiveRFun$new(
			  fun = fun,
			  domain = domain,
			  codomain = codomain, 
			  properties = "noisy",
			  check_values = FALSE
			) 

	# Define termination criterion
	terminator = trm("none")

	# Create optimization instance
	inst = OptimInstanceMultiCrit$new(
	  objective = obj, 
	  search_space = domain,
	  terminator = terminator
	  )

    tuner = mlr3tuning::tnr("sh", eta = 2, n = 2^5, r = 2, mo_method = "indicator_based", np = 5)

    tuner$optimize(inst)
})