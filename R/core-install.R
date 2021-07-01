#' Install Neural Prophet
#'
#' Installs `Neural Prophet` Deep Learning Time Series Forecasting Software
#' using `reticulate::py_install()`.
#' - A `Python` Environment will be created
#' named `nprophet`.
#' - The neuralprophet R package will connect to the `nprophet` Python environment
#'
#' @details
#'
#' __Options for Connecting to Python__
#'
#' - __Recommended__ _Use Pre-Configured Python Environment:_ Use `install_nprophet()` to
#'    install Neural Prophet Python Libraries into a conda environment named 'nprophet'.
#' - __Advanced__ _Use a Custom Python Environment:_ Before running `library(neuralprohet)`,
#'    use `Sys.setenv(NPROPHET_PYTHON = 'path/to/python')` to set the path of your
#'    python executable in an environment that has 'pillow', 'torch', 'numpy', 'pandas',
#'    and 'neuralprophet' available as dependencies.
#'
#' __Package Manager Support (Python Environment)__
#'
#' - __Conda Environments:__ Currently, `install_nprophet()` supports Conda and Miniconda Environments.
#'
#' - __Virtual Environments:__ are not currently supported with the default installation method, `install_nprophet()`.
#'    However, you can connect to virtual environment that you have created using
#'     `Sys.setenv(NPROPHET_PYTHON = 'path/to/python')` prior to running `library(neuralprohet)`.
#'
#' @examples
#' \dontrun{
#' install_nprophet()
#' }
#'
#'
#' @export
install_nprophet <- function() {

    if (!check_conda()) {
        return()
    }

    method <- "conda"

    default_pkgs <- c(
        "pillow==8.3.0",
        "matplotlib==3.4.2",
        "numpy==1.21.0",
        "pandas==1.2.5",
        "torch==1.9.0",
        "neuralprophet==0.2.7"
    )

    #reticulate::conda_create("nprophet", python_version = "3.7.7")

    #os <- reticulate::import("os")

    #os$system("pip install torch==1.6.0 -f https://download.pytorch.org/whl/torch_stable.html")

    cli::cli_process_start("Installing NeuralProphet python dependencies...")
    message("\n")
    reticulate::py_install(
        packages       = default_pkgs,
        envname        = "nprophet",
        method         = method,
        conda          = "auto",
        python_version = "3.7.7",
        pip            = TRUE
    )

    if (!is.null(detect_default_nprophet_env())) {
        cli::cli_process_done(msg_done = "The {.field nprophet} conda environment has been created.")
        cli::cli_alert_info("Please restart your R Session and run {.code library(neuralprophet)} to activate the {.field nprophet} environment.")
    } else {
        cli::cli_process_failed(msg_failed = "The {.field nprophet} conda environment could not be created.")
    }

}


check_conda <- function() {

    conda_list_nrow <- nrow(reticulate::conda_list())

    if (is.null(conda_list_nrow) || conda_list_nrow == 0L) {
        # No conda
        message("Could not detect Conda or Miniconda Package Managers, one of which is required for 'install_nprophet()'. \nAvailable options:\n",
                " - [Preferred] You can install Miniconda (light-weight) using 'reticulate::install_miniconda()'. \n",
                " - Or, you can install the full Aniconda distribution (1000+ packages) using 'reticulate::conda_install()'. \n\n",
                "Then use 'install_nprophet()' to set up the NeuralProphet python environment.")
        conda_found <- FALSE
    } else {
        conda_found <- TRUE
    }

    return(conda_found)
}

