# NEURAL PROPHET TEST ----
context("Test Neural Prophet")

md10 <- m4_daily %>% filter(id == "D10") %>% select(date, value)

splits <- initial_time_split(md10)

# MODEL FITTING ----

test_that("Neural Prophet: model fitting", {

    skip_if_no_nprophet()

    # Model Spec
    model_spec <<- neural_prophet(
        freq                    = "D"
    ) %>%
        set_engine("prophet")

    # ** MODEL FIT

    # Model Fit
    model_fit <- model_spec %>%
        fit(value ~ date, data = training(splits))

    # Test print
    expect_equal(print(model_fit), model_fit)

    # Structure

    testthat::expect_s3_class(model_fit$fit, "neural_prophet_fit_impl")

    testthat::expect_s3_class(model_fit$fit$data, "tbl_df")

    testthat::expect_equal(names(model_fit$fit$data)[1], "date")
    testthat::expect_equal(names(model_fit$fit$extras$components)[1], "ds")
    testthat::expect_equal(names(model_fit$fit$extras$components)[6], "season_weekly")

    # $preproc

    testthat::expect_equal(model_fit$preproc$y_var, "value")


    # ** PREDICTIONS

    # Predictions
    predictions_tbl <- model_fit %>%
        modeltime_calibrate(testing(splits)) %>%
        modeltime_forecast(new_data = testing(splits))

    # Structure
    testthat::expect_identical(nrow(testing(splits)), nrow(predictions_tbl))
    testthat::expect_identical(testing(splits)$date, predictions_tbl$.index)



})

