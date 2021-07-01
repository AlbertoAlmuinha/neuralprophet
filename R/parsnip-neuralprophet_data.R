# These functions are tested indirectly when the models are used. Since this
# function is executed on package startup, you can't execute them to test since
# they are already in the parsnip model database. We'll exclude them from
# coverage stats for this reason.

# nocov start


make_neuralprophet <- function() {

    parsnip::set_new_model("neural_prophet")
    parsnip::set_model_mode("neural_prophet", "regression")

    # prophet_catboost ----

    model <- "neural_prophet"
    engine <- "prophet"

    # * Model ----
    parsnip::set_model_engine(model, mode = "regression", eng = engine)
    parsnip::set_dependency(model, engine, "neuralprophet")
    parsnip::set_dependency(model, engine, "reticulate")

    # * Args - Prophet ----
    parsnip::set_model_arg(
        model        = model,
        eng          = engine,
        parsnip      = "growth",
        original     = "growth",
        func         = list(pkg = "modeltime", fun = "growth"),
        has_submodel = FALSE
    )

    parsnip::set_model_arg(
        model        = model,
        eng          = engine,
        parsnip      = "user_changepoints",
        original     = "changepoints",
        func         = list(pkg = "neuralprophet", fun = "user_changepoints"),
        has_submodel = FALSE
    )

    parsnip::set_model_arg(
        model        = model,
        eng          = engine,
        parsnip      = "changepoint_num",
        original     = "n_changepoints",
        func         = list(pkg = "modeltime", fun = "changepoint_num"),
        has_submodel = FALSE
    )

    parsnip::set_model_arg(
        model        = model,
        eng          = engine,
        parsnip      = "changepoints_range",
        original     = "changepoints_range",
        func         = list(pkg = "modeltime", fun = "changepoints_range"),
        has_submodel = FALSE
    )

    parsnip::set_model_arg(
        model        = model,
        eng          = engine,
        parsnip      = "seasonality_yearly",
        original     = "yearly_seasonality",
        func         = list(pkg = "modeltime", fun = "seasonality_yearly"),
        has_submodel = FALSE
    )

    parsnip::set_model_arg(
        model        = model,
        eng          = engine,
        parsnip      = "seasonality_weekly",
        original     = "weekly_seasonality",
        func         = list(pkg = "modeltime", fun = "seasonality_weekly"),
        has_submodel = FALSE
    )

    parsnip::set_model_arg(
        model        = model,
        eng          = engine,
        parsnip      = "seasonality_daily",
        original     = "daily_seasonality",
        func         = list(pkg = "modeltime", fun = "seasonality_daily"),
        has_submodel = FALSE
    )

    parsnip::set_model_arg(
        model        = model,
        eng          = engine,
        parsnip      = "season",
        original     = "seasonality_mode",
        func         = list(pkg = "modeltime", fun = "season"),
        has_submodel = FALSE
    )

    parsnip::set_model_arg(
        model        = model,
        eng          = engine,
        parsnip      = "trend_reg",
        original     = "trend_reg",
        func         = list(pkg = "neuralprophet", fun = "trend_reg"),
        has_submodel = FALSE
    )

    parsnip::set_model_arg(
        model        = model,
        eng          = engine,
        parsnip      = "trend_reg_threshold",
        original     = "trend_reg_threshold",
        func         = list(pkg = "neuralprophet", fun = "trend_reg_threshold"),
        has_submodel = FALSE
    )

    parsnip::set_model_arg(
        model        = model,
        eng          = engine,
        parsnip      = "num_hidden_layers",
        original     = "num_hidden_layers",
        func         = list(pkg = "neuralprophet", fun = "num_hidden_layers"),
        has_submodel = FALSE
    )

    parsnip::set_model_arg(
        model        = model,
        eng          = engine,
        parsnip      = "d_hidden",
        original     = "d_hidden",
        func         = list(pkg = "neuralprophet", fun = "d_hidden"),
        has_submodel = FALSE
    )

    parsnip::set_model_arg(
        model        = model,
        eng          = engine,
        parsnip      = "ar_sparsity",
        original     = "ar_sparsity",
        func         = list(pkg = "neuralprophet", fun = "ar_sparsity"),
        has_submodel = FALSE
    )

    parsnip::set_model_arg(
        model        = model,
        eng          = engine,
        parsnip      = "learn_rate",
        original     = "learning_rate",
        func         = list(pkg = "dials", fun = "learn_rate"),
        has_submodel = FALSE
    )

    parsnip::set_model_arg(
        model        = model,
        eng          = engine,
        parsnip      = "epochs",
        original     = "epochs",
        func         = list(pkg = "neuralprophet", fun = "epochs"),
        has_submodel = FALSE
    )

    parsnip::set_model_arg(
        model        = model,
        eng          = engine,
        parsnip      = "batch_size",
        original     = "batch_size",
        func         = list(pkg = "neuralprophet", fun = "batch_size"),
        has_submodel = FALSE
    )

    parsnip::set_model_arg(
        model        = model,
        eng          = engine,
        parsnip      = "loss_func",
        original     = "loss_func",
        func         = list(pkg = "neuralprophet", fun = "loss_func"),
        has_submodel = FALSE
    )

    parsnip::set_model_arg(
        model        = model,
        eng          = engine,
        parsnip      = "train_speed",
        original     = "train_speed",
        func         = list(pkg = "neuralprophet", fun = "train_speed"),
        has_submodel = FALSE
    )

    parsnip::set_model_arg(
        model        = model,
        eng          = engine,
        parsnip      = "normalize_y",
        original     = "normalize_y",
        func         = list(pkg = "neuralprophet", fun = "normalize_y"),
        has_submodel = FALSE
    )

    parsnip::set_model_arg(
        model        = model,
        eng          = engine,
        parsnip      = "impute_missing",
        original     = "impute_missing",
        func         = list(pkg = "neuralprophet", fun = "impute_missing"),
        has_submodel = FALSE
    )

    parsnip::set_model_arg(
        model        = model,
        eng          = engine,
        parsnip      = "n_forecasts",
        original     = "n_forecasts",
        func         = list(pkg = "neuralprophet", fun = "n_forecasts"),
        has_submodel = FALSE
    )

    parsnip::set_model_arg(
        model        = model,
        eng          = engine,
        parsnip      = "n_lags",
        original     = "n_lags",
        func         = list(pkg = "neuralprophet", fun = "n_lags"),
        has_submodel = FALSE
    )

    parsnip::set_model_arg(
        model        = model,
        eng          = engine,
        parsnip      = "freq",
        original     = "freq",
        func         = list(pkg = "neuralprophet", fun = "freq"),
        has_submodel = FALSE
    )





    # * Encoding ----
    parsnip::set_encoding(
        model   = model,
        eng     = engine,
        mode    = "regression",
        options = list(
            predictor_indicators = "none",
            compute_intercept    = FALSE,
            remove_intercept     = FALSE,
            allow_sparse_x       = FALSE
        )
    )

    # * Fit ----
    parsnip::set_fit(
        model         = model,
        eng           = engine,
        mode          = "regression",
        value         = list(
            interface = "data.frame",
            protect   = c("x", "y"),
            func      = c(fun = "neural_prophet_fit_impl"),
            defaults  = list()
        )
    )

    # * Predict ----
    parsnip::set_pred(
        model         = model,
        eng           = engine,
        mode          = "regression",
        type          = "numeric",
        value         = list(
            pre       = NULL,
            post      = NULL,
            func      = c(fun = "predict"),
            args      =
                list(
                    object   = rlang::expr(object$fit),
                    new_data = rlang::expr(new_data)
                )
        )
    )

}

# nocov end
