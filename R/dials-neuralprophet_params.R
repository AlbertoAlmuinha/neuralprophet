#' Tuning Parameters for Neural Prophet Models
#'
#'
#' @inheritParams dials::Laplace
#'
#' @details
#' The main parameters for Neural Prophet models are:
#'
#'  - `trend_reg`: the trend rate changes can be regularized by setting trend_reg to a value greater zero.
#'  This is a useful feature that can be used to automatically detect relevant changepoints.
#'  - `trend_reg_threshold`: Threshold for the trend regularization
#'  - `num_hidden_layers`: num_hidden_layers defines the number of hidden layers of the FFNNs used in the overall model.
#'  - `d_hidden`: d_hidden is the number of units in the hidden layers.
#'  - `ar_sparsity`: For ar_sparsity values in the range 0-1 are expected with 0 inducing complete sparsity and 1 imposing no regularization at
#' all
#'
#' @examples
#' trend_reg()
#'
#' num_hidden_layers()
#'
#' ar_sparsity()
#'
#'
#' @name nprophet_params


#' @export
#' @rdname nprophet_params
trend_reg <- function(range = c(0, 100), trans = NULL) {
    dials::new_quant_param(
        type      = "double",
        range     = range,
        inclusive = c(TRUE, TRUE),
        trans     = trans,
        label     = c(trend_reg = "The trend rate changes can be regularized by setting trend_reg to a value greater zero. "),
        finalize  = NULL
    )
}


#' @export
#' @rdname nprophet_params
trend_reg_threshold <- function(range = c(0, 10), trans = NULL) {
    dials::new_quant_param(
        type      = "double",
        range     = range,
        inclusive = c(TRUE, TRUE),
        trans     = trans,
        label     = c(trend_reg_threshold = "Threshold for the trend regularization"),
        finalize  = NULL
    )
}

#' @export
#' @rdname nprophet_params
num_hidden_layers <- function(range = c(0L, 10L), trans = NULL) {
    dials::new_quant_param(
        type      = "integer",
        range     = range,
        inclusive = c(TRUE, TRUE),
        trans     = trans,
        label     = c(num_hidden_layers = "num_hidden_layers defines the number of hidden layers of the FFNNs used in the overall model."),
        finalize  = NULL
    )
}

#' @export
#' @rdname nprophet_params
d_hidden <- function(range = c(0L, 500L), trans = NULL) {
    dials::new_quant_param(
        type      = "integer",
        range     = range,
        inclusive = c(TRUE, TRUE),
        trans     = trans,
        label     = c(d_hidden = "d_hidden is the number of units in the hidden layers."),
        finalize  = NULL
    )
}

#' @export
#' @rdname nprophet_params
ar_sparsity <- function(range = c(0, 1), trans = NULL) {
    dials::new_quant_param(
        type      = "double",
        range     = range,
        inclusive = c(TRUE, TRUE),
        trans     = trans,
        label     = c(ar_sparsity = "For ar_sparsity values in the range 0-1 are expected with 0 inducing complete sparsity and 1 imposing no regularization at all"),
        finalize  = NULL
    )
}

