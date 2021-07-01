#' Neural Prophet Environment Connection Utilities
#'
#' These functions are used for testing and establishing a python environment
#' connection with `neuralprophet`.
#'
#' @details
#'
#' - `is_nprophet_activated()`: Determines if a Neural Prophet Environment has been activated
#'   during `library(neuralprophet)`.
#'     - If `TRUE`, then you should be good to go.
#'     - If `FALSE`, then a connection between `neuralprophet`
#'       and your neuralprophet Python has _not_ been activated.
#'
#' - `activate_nprophet()`: Attempts to activate a connection between `neuralprophet`
#'   and an associated NeuralProphet Python Environment using `reticulate::use_condaenv(required = TRUE)`.
#'     - It first looks for the system environment variable, 'NPROPHET_PYTHON', for a path to the python executable
#'     - It next looks for a Conda Environment named 'nprophet' (this is what most users will have)
#'
#' - `get_python_env()`: Returns the configuration for the python environment that is being discovered
#'   using `reticulate::py_discover_config()`.
#'
#' - `check_nprophet_dependencies()`: Checks whether Neural Prophet required python dependencies are present in the
#'   currently activated Python Environment.
#'
#' - `detect_default_nprophet_env()`: Detects if an 'nprophet' python environment is available.
#'     - Returns a `tibble` containing the
#'     - Returns `NULL` if an 'nprophet' environment is not detected
#'
#' @seealso
#' - [install_nprophet()] - Used to install the python environment needed to run `neuralprophet`.
#'
#'
#' @examples
#' \donttest{
#' # Returns TRUE if NeuralProphet connection established on package load
#' is_nprophet_activated()
#'
#' #
#'
#' }
#'
#'
#'
#' @name nprophet-env

#' @export
#' @rdname nprophet-env
is_nprophet_activated <- function() {
    pkg.env$activated
}

#' @export
#' @rdname nprophet-env
activate_nprophet <- function() {

    # STEP 1 - CHECK FOR GLUONTS_PYTHON
    nprophet_python <- Sys.getenv("NPROPHET_PYTHON", unset = NA)
    custom_env_detected <- !is.na(nprophet_python)
    if (custom_env_detected) {

        # Sys.setenv('RETICULATE_PYTHON' = gluonts_python) # More forceful, generates warning and errors
        reticulate::use_python(python = nprophet_python, required = TRUE)
        pkg.env$activated <- TRUE

    }

    # STEP 2 - CHECK FOR DEFAULT r-gluonts ENV
    default_conda_env <- detect_default_nprophet_env()
    conda_envs_found  <- !is.null(default_conda_env)
    if (all(c(!pkg.env$activated, conda_envs_found))) {

        # Sys.setenv('RETICULATE_PYTHON' = default_conda_env$python[[1]])
        try({
            reticulate::use_python(python = default_conda_env$python[[1]], required = TRUE)
            pkg.env$activated <- TRUE
        }, silent = TRUE)

    }

}

#' @export
#' @rdname nprophet-env
get_python_env <- function() {
    reticulate::py_discover_config()
}

#' @export
#' @rdname nprophet-env
check_nprophet_dependencies <- function() {

    dependencies_ok <- FALSE
    try({
        dependencies_ok <- all(
            reticulate::py_module_available("numpy"),
            reticulate::py_module_available("pandas"),
            reticulate::py_module_available("neuralprophet"),
            reticulate::py_module_available("torch"),
            reticulate::py_module_available("matplotlib"),
            reticulate::py_module_available("PIL")
        )
    }, silent = TRUE)

    return(dependencies_ok)
}

#' @export
#' @rdname nprophet-env
detect_default_nprophet_env <- function() {

    ret <- NULL
    tryCatch({

        ret <- reticulate::conda_list() %>%
            tibble::as_tibble() %>%
            dplyr::filter(stringr::str_detect(python, pkg.env$env_name)) %>%
            dplyr::slice(1)

    }, error = function(e) {
        ret <- NULL
    })

    if (!is.null(ret)) {
        if (nrow(ret) == 0) {
            ret <- NULL
        }
    }

    return(ret)

}
