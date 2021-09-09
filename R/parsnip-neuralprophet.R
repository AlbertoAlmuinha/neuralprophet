# NEURAL PROPHET ----

#' General Interface for Neural Prophet Time Series Models
#'
#' `neural_prophet()` is a way to generate a _specification_ of a NEURAL PROPHET model
#'  before fitting and allows the model to be created using
#'  different packages. Currently the only package is `neuralprophet` from Python through `reticulate`.
#'
#' @inheritParams modeltime::prophet_reg
#' @param user_changepoints If a list of changepoints is supplied, n_changepoints and changepoints_range are ignored. This list is
#' instead used to set the dates at which the trend rate is allowed to change.
#' @param trend_reg  the trend rate changes can be regularized by setting trend_reg to a value greater zero. This is a useful feature that can be used to automatically detect relevant changepoints.
#' @param trend_reg_threshold Threshold for the trend regularization
#' @param seasonality_mode The default seasonality_mode is additive. This means that no heteroscedasticity is expected in the series in
#' terms of the seasonality. However, if the series contains clear variance, where the seasonal fluctuations become
#' larger proportional to the trend, the seasonality_mode can be set to multiplicative.
#' @param num_hidden_layers num_hidden_layers defines the number of hidden layers of the FFNNs used in the overall model.
#' This includes the AR-Net and the FFNN of the lagged regressors. The default is 0, meaning that the FFNNs will have only
#' one final layer of size n_forecasts. Adding more layers results in increased complexity and also increased computational
#' time, consequently. However, the added number of hidden layers can help build more complex relationships especially useful
#' for the lagged regressors. To tradeoff between the computational complexity and the improved accuracy the num_hidden_layers
#' is recommended to be set in between 1-2. Nevertheless, in most cases a good enough performance can be achieved by having
#' no hidden layers at all.
#' @param d_hidden d_hidden is the number of units in the hidden layers. This is only considered if num_hidden_layers
#' is specified, otherwise ignored. The default value for d_hidden if not specified is (n_lags + n_forecasts). If tuned
#' manually, the recommended practice is to set a value in between n_lags and n_forecasts for d_hidden. It is also
#' important to note that with the current implementation, NeuralProphet sets the same d_hidden for the all the hidden
#' layers.
#' @param ar_sparsity NeuralProphet also contains a number of regularization parameters to control the model coefficients
#' and introduce sparsity into the model. This also helps avoid overfitting of the model to the training data.For
#' ar_sparsity values in the range 0-1 are expected with 0 inducing complete sparsity and 1 imposing no regularization at
#' all. ar_sparsity along with n_lags can be used for data exploration and feature selection. You can use a larger number
#' of lags thanks to the scalability of AR-Net and use the scarcity to identify important influence of past time steps on
#' the prediction accuracy.
#' @param learn_rate NeuralProphet is fit with stochastic gradient descent - more precisely, with an AdamW optimizer and
#'  a One-Cycle policy. If the parameter learning_rate is not specified, a learning rate range test is conducted to
#'  determine the optimal learning rate. A number for the rate at which the algorithm adapts from iteration-to-iteration.
#' @param epochs The epochs and the loss_func are two other parameters that directly affect the model training process.
#' If not defined, both are automatically set based on the dataset size. They are set in a manner that controls the
#' total number training steps to be around 1000 to 4000.
#' @param batch_size number of samples that will be propagated through the network
#' @param loss_func The default loss function is the 'Huber' loss, which is considered to be robust to outliers.
#' However, you are free to choose the standard MSE or any other PyTorch torch.nn.modules.loss loss function.
#' @param train_speed Number indicating the speed at which training of the network occurs.
#' @param normalize_y is about scaling the time series before modelling. By default, NeuralProphet performs a (soft)
#' min-max normalization of the time series. Normalization can help the model training process if the series values
#' fluctuate heavily. However, if the series does not such scaling, users can turn this off or select another
#' normalization.
#' @param impute_missing is about imputing the missing values in a given series. S imilar to Prophet, NeuralProphet
#' too can work with missing values when it is in the regression mode without the AR-Net. However, when the
#' autocorrelation needs to be captured, it is necessary for the missing values to be imputed, since then the
#' modelling becomes an ordered problem. Letting this parameter at its default can get the job done perfectly in most
#' cases.
#' @param n_forecasts is the size of the forecast horizon. The default value of 1 means that the model forecasts one
#' step into the future.
#' @param n_lags defines whether the AR-Net is enabled (if n_lags > 0) or not. The value for n_lags is usually
#' recommended to be greater than n_forecasts, if possible since it is preferable for the FFNNs to encounter at
#' least n_forecasts length of the past in order to predict n_forecasts into the future. Thus, n_lags determine
#' how far into the past the auto-regressive dependencies should be considered. This could be a value chosen based
#' on either domain expertise or an empirical analysis.
#' @param freq A pandas timeseries frequency such as "5min" for 5-minutes or "D" for daily. Refer to Pandas Offset Aliases
#'
#' @details
#' The data given to the function are not saved and are only used
#'  to determine the _mode_ of the model. For `neural_prophet()`, the
#'  mode will always be "regression".
#'
#' The model can be created using the `fit()` function using the
#'  following _engines_:
#'
#'  - "prophet" (default) - Connects to neuralprophet.NeuralProphet() (Python)
#'
#' __Main Arguments__
#'
#' The main arguments (tuning parameters) for the __NEURAL PROPHET__ model are:
#'
#' - `growth`: String 'linear' or 'logistic' to specify a linear or logistic trend.
#' - `changepoint_num`: Number of potential changepoints to include for modeling trend.
#' - `changepoint_range`: Range changepoints that adjusts how close to the end
#'    the last changepoint can be located.
#' - `season`: 'additive' (default) or 'multiplicative'.
#' - `ar_sparsity`: For ar_sparsity values in the range 0-1 are expected with 0 inducing complete sparsity and 1 imposing no regularization at
#' all
#' - `num_hidden_layers`: num_hidden_layers defines the number of hidden layers of the FFNNs used in the overall model.
#' - `d_hidden`: d_hidden is the number of units in the hidden layers.
#' - `trend_reg`: the trend rate changes can be regularized by setting trend_reg to a value greater zero.
#'  This is a useful feature that can be used to automatically detect relevant changepoints.
#'
#'
#'
#' These arguments are converted to their specific names at the
#'  time that the model is fit.
#'
#' Other options and argument can be
#'  set using `set_engine()` (See Engine Details below).
#'
#' If parameters need to be modified, `update()` can be used
#'  in lieu of recreating the object from scratch.
#'
#'
#' @section Engine Details:
#'
#' The standardized parameter names in `neuralprophet` can be mapped to their original
#' names in each engine.
#' Other options can be set using `set_engine()`.
#'
#'
#' __prophet__
#'
#' Limitations:
#'
#' - `prophet::add_seasonality()` is not currently implemented. It's used to
#'  specify non-standard seasonalities using fourier series. An alternative is to use
#'  `step_fourier()` and supply custom seasonalities as Extra Regressors.

#' __Date and Date-Time Variable__
#'
#' It's a requirement to have a date or date-time variable as a predictor.
#' The `fit()` interface accepts date and date-time features and handles them internally.
#'
#' - `fit(y ~ date)`
#'
#'
#' __Univariate (No Extra Regressors):__
#'
#' For univariate analysis, you must include a date or date-time feature. Simply use:
#'
#'  - Formula Interface (recommended): `fit(y ~ date)` will ignore xreg's.
#'
#' __Events__
#'
#' To include events correctly, the following conditions must be met:
#'
#' - Event variable names must contain `events` in their name. For example: "events_one", "events_two".
#'
#' - Pass a list called `add_events` through set_engine(). This list will define the characteristics of our events.
#' It should contain the elements contained in the `add_events` method in Python.
#'
#' - Include events in the formula as external regressors
#'
#' _Example:_
#'
#'  neural_prophet(freq = "D") %>% set_engine(add_events = list(events = c("events_1", "events_2"),
#'                                                              regularization = 0.5)) %>%
#'      fit(y ~ date + events_1 + events_2, data = df)
#'
#' __Future Regressor__
#'
#' To include Future Regressors correctly, the following conditions must be met:
#'
#' - Future Regressors variable names must contain `future_` in their name. For example: "future_one", "future_two".
#'
#' - Any columns that aren't labeled "event_" or "lagged_" are added as Future Regressors (except date one).
#'
#' - Pass a list called `add_future_regressor` through set_engine(). This list will define the characteristics of our future_regressors
#' It should contain the elements contained in the `add_future_regressor` method in Python.
#'
#' - Include future_regressors in the formula as external regressors
#'
#' _Example:_
#'
#'  neural_prophet(freq = "D") %>% set_engine(add_future_regressor = list(name = c("future_1", "future_2"),
#'                                                                        regularization = 0.5)) %>%
#'      fit(y ~ date + future_1 + future_2, data = df)
#'
#'
#' __Lagged Regressor__
#'
#' To include Lagged Regressors correctly, the following conditions must be met:
#'
#' - Lagged Regressors variable names must contain `lagged` in their names. For example: "lagged_one", "lagged_two".
#'
#' - Pass a list called `add_lagged_regressor` through set_engine(). This list will define the characteristics of our lagged_regressor
#' It should contain the elements contained in the `add_lagged_regressor` method in Python.
#'
#' - Include lagged regressors in the formula as external regressors
#'
#' _Example:_
#'
#'  neural_prophet(freq = "D") %>% set_engine(add_lagged_regressor = list(name = c("lagged_1", "lagged_2"),
#'                                                                        regularization = 0.5)) %>%
#'      fit(y ~ date + lagged_1 + lagged_2, data = df)
#'
#'
#' @seealso [fit.model_spec()], [set_engine()]
#'
#' @examples
#' library(dplyr)
#' library(lubridate)
#' library(parsnip)
#' library(rsample)
#' library(timetk)
#'
#' # Data
#' md10 <- m4_daily %>% filter(id == "D10")
#' md10
#'
#' # Split Data 80/20
#' splits <- initial_time_split(md10, prop = 0.8)
#'
#' # ---- NEURAL PROPHET ----
#'
#' # Model Spec
#' model_spec <- neural_prophet(
#'     freq = "D"
#' ) %>%
#'     set_engine("prophet")
#'
#' # Fit Spec
#' model_fit <- model_spec %>%
#'     fit(log(value) ~ date,
#'         data = training(splits))
#' model_fit
#'
#' @export
neural_prophet <- function(mode = "regression", growth = NULL, user_changepoints = NULL, changepoint_num = NULL, changepoint_range = NULL,
                          seasonality_yearly = NULL, seasonality_weekly = NULL, seasonality_daily = NULL,
                          season = NULL, trend_reg = NULL, trend_reg_threshold = NULL, seasonality_mode = NULL,
                          num_hidden_layers = NULL, d_hidden = NULL, ar_sparsity = NULL, learn_rate = NULL,
                          epochs = NULL, batch_size = NULL, loss_func = NULL, train_speed = NULL, normalize_y = NULL,
                          impute_missing = NULL, n_forecasts = NULL, n_lags = NULL, freq = NULL) {

    args <- list(

        # Prophet
        growth                    = rlang::enquo(growth),
        user_changepoints         = rlang::enquo(user_changepoints),
        changepoint_num           = rlang::enquo(changepoint_num),
        changepoint_range         = rlang::enquo(changepoint_range),
        seasonality_yearly        = rlang::enquo(seasonality_yearly),
        seasonality_weekly        = rlang::enquo(seasonality_weekly),
        seasonality_daily         = rlang::enquo(seasonality_daily),
        season                    = rlang::enquo(season),
        trend_reg                 = rlang::enquo(trend_reg),
        trend_reg_threshold       = rlang::enquo(trend_reg_threshold),
        seasonality_mode          = rlang::enquo(seasonality_mode),
        num_hidden_layers         = rlang::enquo(num_hidden_layers),
        d_hidden                  = rlang::enquo(d_hidden),
        ar_sparsity               = rlang::enquo(ar_sparsity),
        learn_rate                = rlang::enquo(learn_rate),
        epochs                    = rlang::enquo(epochs),
        batch_size                = rlang::enquo(batch_size),
        loss_func                 = rlang::enquo(loss_func),
        train_speed               = rlang::enquo(train_speed),
        normalize_y               = rlang::enquo(normalize_y),
        impute_missing            = rlang::enquo(impute_missing),
        n_forecasts               = rlang::enquo(n_forecasts),
        n_lags                    = rlang::enquo(n_lags),
        freq                      = rlang::enquo(freq)
    )

    parsnip::new_model_spec(
        "neural_prophet",
        args     = args,
        eng_args = NULL,
        mode     = mode,
        method   = NULL,
        engine   = NULL
    )

}

#' @export
print.neural_prophet <- function(x, ...) {
    cat("Neural Prophet (", x$mode, ")\n\n", sep = "")
    parsnip::model_printer(x, ...)

    if(!is.null(x$method$fit$args)) {
        cat("Model fit template:\n")
        print(parsnip::show_call(x))
    }

    invisible(x)
}

#' @export
#' @importFrom stats update
update.neural_prophet <- function(object,
                               parameters = NULL, growth = NULL,user_changepoints = NULL, changepoint_num = NULL, changepoint_range = NULL,
                               seasonality_yearly = NULL, seasonality_weekly = NULL, seasonality_daily = NULL,
                               season = NULL, trend_reg = NULL, trend_reg_threshold = NULL, seasonality_mode = NULL,
                               num_hidden_layers = NULL, d_hidden = NULL, ar_sparsity = NULL, learn_rate = NULL,
                               epochs = NULL, batch_size = NULL, loss_func = NULL, train_speed = NULL, normalize_y = NULL,
                               impute_missing = NULL, n_forecasts = NULL, n_lags = NULL, freq = NULL, fresh = FALSE, ...) {

    parsnip::update_dot_check(...)

    if (!is.null(parameters)) {
        parameters <- parsnip::check_final_param(parameters)
    }

    args <- list(

        # Prophet
        growth                    = rlang::enquo(growth),
        user_changepoints         = rlang::enquo(user_changepoints),
        changepoint_num           = rlang::enquo(changepoint_num),
        changepoint_range         = rlang::enquo(changepoint_range),
        seasonality_yearly        = rlang::enquo(seasonality_yearly),
        seasonality_weekly        = rlang::enquo(seasonality_weekly),
        seasonality_daily         = rlang::enquo(seasonality_daily),
        season                    = rlang::enquo(season),
        trend_reg                 = rlang::enquo(trend_reg),
        trend_reg_threshold       = rlang::enquo(trend_reg_threshold),
        seasonality_mode          = rlang::enquo(seasonality_mode),
        num_hidden_layers         = rlang::enquo(num_hidden_layers),
        d_hidden                  = rlang::enquo(d_hidden),
        ar_sparsity               = rlang::enquo(ar_sparsity),
        learn_rate                = rlang::enquo(learn_rate),
        epochs                    = rlang::enquo(epochs),
        batch_size                = rlang::enquo(batch_size),
        loss_func                 = rlang::enquo(loss_func),
        train_speed               = rlang::enquo(train_speed),
        normalize_y               = rlang::enquo(normalize_y),
        impute_missing            = rlang::enquo(impute_missing),
        n_forecasts               = rlang::enquo(n_forecasts),
        n_lags                    = rlang::enquo(n_lags),
        freq                      = rlang::enquo(freq)
    )

    args <- parsnip::update_main_parameters(args, parameters)

    if (fresh) {
        object$args <- args
    } else {
        null_args <- purrr::map_lgl(args, parsnip::null_value)
        if (any(null_args))
            args <- args[!null_args]
        if (length(args) > 0)
            object$args[names(args)] <- args
    }

    parsnip::new_model_spec(
        "neural_prophet",
        args     = object$args,
        eng_args = object$eng_args,
        mode     = object$mode,
        method   = NULL,
        engine   = object$engine
    )
}


#' @export
#' @importFrom parsnip translate
translate.neural_prophet <- function(x, engine = x$engine, ...) {
    if (is.null(engine)) {
        message("Used `engine = 'prophet'` for translation.")
        engine <- "prophet"
    }
    x <- parsnip::translate.default(x, engine, ...)

    x
}



# FIT BRIDGE - PROPHET ----

#' Bridge Prophet-Catboost Modeling function
#'
#' @inheritParams neural_prophet
#' @param formula Value
#' @param data Value
#' @param changepoints changepoints
#' @param n_changepoints n_changepoints
#' @param changepoints_range changepoints_range
#' @param yearly_seasonality yearly_seasonality
#' @param weekly_seasonality weekly_seasonality
#' @param daily_seasonality daily_seasonality
#' @param seasonality_reg seasonality_reg
#' @param learning_rate learning_rate
#' @param ... Additional arguments
#'
#' @export
#' @importFrom stats frequency
neural_prophet_fit_impl <- function(formula, data,
                                  growth = "linear",
                                  changepoints = NULL,
                                  n_changepoints = 5L,
                                  changepoints_range = 0.8,
                                  yearly_seasonality = "auto",
                                  weekly_seasonality = "auto",
                                  daily_seasonality = "auto",
                                  seasonality_mode = "additive",
                                  trend_reg = 0L,
                                  trend_reg_threshold = FALSE,
                                  seasonality_reg = 0L,
                                  n_forecasts = 1L,
                                  n_lags = 0L,
                                  num_hidden_layers = 0L,
                                  d_hidden = NULL,
                                  ar_sparsity = NULL,
                                  learning_rate = NULL,
                                  epochs = NULL,
                                  batch_size = NULL,
                                  loss_func = "Huber",
                                  train_speed = NULL,
                                  normalize_y = "auto",
                                  impute_missing = TRUE,
                                  freq = NULL,
                                  ...) {

    args <- list(...)

    d_hidden            <- if (is.null(d_hidden)) {reticulate::py_none()} else d_hidden
    ar_sparsity         <- if (is.null(ar_sparsity)) {reticulate::py_none()} else ar_sparsity
    learning_rate       <- if (is.null(learning_rate)) {reticulate::py_none()} else learning_rate
    epochs              <- if (is.null(epochs)) {reticulate::py_none()} else as.integer(epochs)
    batch_size          <- if (is.null(batch_size)) {reticulate::py_none()} else as.integer(batch_size)
    train_speed         <- if (is.null(train_speed)) {reticulate::py_none()} else train_speed

    changepoints        <- if (!is.null(changepoints)) {reticulate::r_to_py(changepoints)}
    n_changepoints      <- reticulate::r_to_py(as.integer(n_changepoints))
    changepoints_range  <- reticulate::r_to_py(changepoints_range)
    trend_reg           <- reticulate::r_to_py(as.integer(trend_reg))
    trend_reg_threshold <- if (!is.null(trend_reg_threshold)) {reticulate::r_to_py(trend_reg_threshold)}
    seasonality_reg     <- reticulate::r_to_py(as.integer(seasonality_reg))
    n_forecasts         <- reticulate::r_to_py(as.integer(n_forecasts))
    n_lags              <- reticulate::r_to_py(as.integer(n_lags))
    num_hidden_layers   <- reticulate::r_to_py(as.integer(num_hidden_layers))
    #d_hidden            <- if (d_hidden != reticulate::py_none()) {reticulate::r_to_py(as.integer(d_hidden))}
    #ar_sparsity         <- if (ar_sparsity != reticulate::py_none()) {reticulate::r_to_py(ar_sparsity)}
    #learning_rate       <- if (learning_rate != reticulate::py_none()) {reticulate::r_to_py(learning_rate)}
    #epochs              <- if (epochs != reticulate::py_none()) {reticulate::r_to_py(as.integer(epochs))}
    #batch_size          <- if (batch_size != reticulate::py_none()) {reticulate::r_to_py(as.integer(batch_size))}

    y <- all.vars(formula)[1]
    x <- attr(stats::terms(formula, data = data), "term.labels")

    # X & Y
    # Expect outcomes  = vector
    # Expect predictor = data.frame
    outcome <- data[[y]]
    predictors <- data %>% dplyr::select(dplyr::all_of(x))

    growth <- stringr::str_to_lower(growth)
    seasonality_mode <- stringr::str_to_lower(seasonality_mode)

    if (!growth[1] %in% c("linear", "logistic")) {
        cli::cli_alert_info("growth must be linear or logistic. Using 'linear'...")
        growth <- 'linear'
    }

    if (!seasonality_mode[1] %in% c("additive", "multiplicative")) {
        cli::cli_alert_info("seasonality_mode must be 'additive' or 'multiplicative'. Using 'additive'...")
        seasonality_mode <- 'additive'
    }

    # INDEX & PERIOD
    # Determine Period, Index Col, and Index
    index_tbl <- modeltime::parse_index_from_data(predictors)
    idx_col   <- names(index_tbl)
    idx       <- timetk::tk_index(index_tbl)

    # XREGS
    # Clean names, get xreg recipe, process predictors
    xreg_recipe <- modeltime::create_xreg_recipe(predictors, prepare = TRUE)
    xreg_tbl    <- modeltime::juice_xreg_recipe(xreg_recipe, format = "tbl")

    # FIT

    # Construct Data Frame
    df <- tibble::tibble(
        ds = idx,
        y  = outcome
    )

    # Construct model
    # Fit model
    fit_prophet <- pkg.env$nprophet$NeuralProphet(
        growth = growth,
        changepoints = changepoints,
        n_changepoints = n_changepoints,
        changepoints_range = changepoints_range,
        yearly_seasonality = yearly_seasonality,
        weekly_seasonality = weekly_seasonality,
        daily_seasonality = daily_seasonality,
        seasonality_mode = seasonality_mode,
        trend_reg = trend_reg,
        trend_reg_threshold = trend_reg_threshold,
        seasonality_reg = seasonality_reg,
        n_forecasts = n_forecasts,
        n_lags = n_lags,
        num_hidden_layers = num_hidden_layers,
        d_hidden = d_hidden,
        ar_sparsity = ar_sparsity,
        learning_rate = learning_rate,
        epochs = epochs,
        batch_size = batch_size,
        loss_func = loss_func,
        train_speed = train_speed,
        #normalize_y = normalize_y,
        impute_missing = impute_missing
    )

    if (any(names(args) == "add_events")){

        df <- df %>% dplyr::bind_cols(xreg_tbl %>% dplyr::select(dplyr::contains("events")))

        rlang::exec(fit_prophet$add_events, !!!args$add_events)

        events_df <- split_events(df) %>% split_convert()

    }

    if (any(names(args) == "add_lagged_regressor")){

        df <- df %>% dplyr::bind_cols(xreg_tbl %>% dplyr::select(dplyr::contains("lagged")))

        rlang::exec(fit_prophet$add_lagged_regressor, !!!args$add_lagged_regressor)

    }

    if (length(xreg_tbl)>0){

        df <- df %>% dplyr::bind_cols(xreg_tbl %>% dplyr::select(-dplyr::all_of(dplyr::contains("events")),
                                                                 -dplyr::all_of(dplyr::contains("lagged"))))

        if (any(names(args) == "add_future_regressor")){
            args$add_future_regressor$names <- names(df)[3:length(names(df))]
        } else {
            args["add_future_regressor"] <- list(names = names(df)[3:length(names(df))])
        }

        rlang::exec(fit_prophet$add_future_regressor, !!!args$add_future_regressor)

        regressors_df <- df[, names(df)[3:length(names(df))]]

    }



    if (any(names(args) == "add_future_regressor") & any(names(args) == "add_events")) {
        val <- "1"
    }

    if (any(names(args) == "add_future_regressor") & !any(names(args) == "add_events")) {
        val <- "2"
    }

    if (!any(names(args) == "add_future_regressor") & any(names(args) == "add_events")) {
        val <- "3"
    }

    if (!any(names(args) == "add_future_regressor") & !any(names(args) == "add_events")) {
        val <- "4"
    }

    metrics <- fit_prophet$fit(df, freq = freq)

    future_df <- switch(val,
                        "1" = fit_prophet$make_future_dataframe(df, n_historic_predictions = dim(df)[1], events_df = events_df, regressors_df = regressors_df),
                        "2" = fit_prophet$make_future_dataframe(df, n_historic_predictions = dim(df)[1], regressors_df = regressors_df),
                        "3" = fit_prophet$make_future_dataframe(df, n_historic_predictions = dim(df)[1], events_df = events_df),
                        "4" = fit_prophet$make_future_dataframe(df, n_historic_predictions = dim(df)[1]))


    preds <- fit_prophet$predict(future_df) %>% reticulate::py_to_r() %>% tibble::as_tibble() %>% dplyr::slice_max(n = dim(df)[1], order_by = dplyr::desc(ds))

    if (reticulate::py_to_r(n_lags) > 0){
        preds <- preds[(reticulate::py_to_r(n_lags)+1):dim(df)[1],]
    }


    # RETURN A NEW MODELTIME BRIDGE

    # Class - Add a class for the model
    class <- "neural_prophet_fit_impl"

    # Models - Insert model_1 and model_2 into a list
    models <- list(
        model_1 = fit_prophet
    )

    # Data - Start with index tbl and add .actual, .fitted, and .residuals columns
    data <- preds %>% dplyr::select(ds) %>%
        dplyr::mutate(
            .actual    =  preds$y,
            .fitted    =  preds$yhat1,
            .residuals = preds$residual1
        )

    # Extras - Pass on transformation recipe
    extras <- list(
        components  = preds,
        xreg_recipe = xreg_recipe,
        args        = args,
        date_col    = idx_col,
        future_df   = future_df,
        n_lags      = reticulate::py_to_r(n_lags),
        n_forecasts = reticulate::py_to_r(n_forecasts),
        value       = y
    )

    # Model Description - Gets printed to describe the high-level model structure
    desc <- paste0("Neural Prophet")

    # Create new model
    modeltime::new_modeltime_bridge(
        class  = class,
        models = models,
        data   = data,
        extras = extras,
        desc   = desc
    )
}

#' @export
print.neural_prophet_fit_impl <- function(x, ...) {

    if (!is.null(x$desc)) cat(paste0(x$desc,"\n"))
    cat("Model: Neural Prophet\n")
    print(x$models$model_1)
    invisible(x)
}


# PREDICT BRIDGE ----

#' @export
predict.neural_prophet_fit_impl <- function(object, new_data, ...) {
    neural_prophet_predict_impl(object, new_data, ...)
}


#' Bridge prediction Function for NEURAL PROPHET Models
#'
#' @inheritParams parsnip::predict.model_fit
#' @param ... Additional arguments passed to `neuralprophet::NeuralProphet()`
#'
#' @export
neural_prophet_predict_impl <- function(object, new_data, ...) {

    model           <- object$models$model_1
    args            <- object$extras$args
    date_col        <- object$extras$date_col
    xreg_recipe     <- object$extras$xreg_recipe
    n_lags          <- object$extras$n_lags
    n_forecasts     <- object$extras$n_forecasts
    value_col       <- object$extras$value

    new_data1 <- new_data  %>% dplyr::rename(ds = date_col, y = value_col)

    xreg_tbl <- modeltime::bake_xreg_recipe(xreg_recipe, new_data, format = "tbl")


    if (any(names(args) == "add_events")){

        new_data_events <- new_data1 %>% dplyr::select(ds, dplyr::contains("events"))

        events_future_df <- split_events(new_data_events) %>% split_convert()

    }

    if (length(xreg_tbl)>0){
        regressors_future_df <- xreg_tbl %>% dplyr::select(-dplyr::all_of(dplyr::contains("events")), -dplyr::all_of(dplyr::contains("lagged")))
    } else {
        regressors_future_df <- NULL
    }

    val <- get_combination_value(args)

    # Construct Future Frame
    future_df <- get_combination_df(new_data1, val, date_col, model, events_future_df, regressors_future_df)

    # PREDICTIONS
    preds_prophet_df <- model$predict(future_df) %>%
                         reticulate::py_to_r() %>%
                         tibble::as_tibble() %>%
                         dplyr::slice_max(n = dim(future_df)[1]-1, order_by = dplyr::desc(ds))

    convert_to_number <- function(x){
        if (is.null(x)){99999} else {x}
    }


    # Return predictions as numeric vector
    if (n_lags > 0){
        preds_prophet_df <- preds_prophet_df$yhat1 %>% purrr::map_dbl(convert_to_number)
        preds_prophet <- preds_prophet_df[(n_lags+1):(length(preds_prophet_df)-n_forecasts+2)]

        preds_prophet <- c(rep(NA, n_lags-1), preds_prophet)

    } else {
        preds_prophet <- preds_prophet_df %>% dplyr::pull(yhat1)
    }

    return(preds_prophet)

}
