
# PACKAGE IMPORTS ----

#' @import modeltime
#' @importFrom reticulate py


# ONLOAD UTILITIES ----

msg_no_nprophet <- function() {
    cli::cli_h1("Python Dependency Check {.pkg neuralprophet}")
    cli::cli_alert_danger('Neural Prophet Python Dependencies Not Found')
    cli::cli_alert_info("Available Options: ")
    cli::cli_ol(id = "nprophet_installation_options")
    cli::cli_li("{.strong [Option 1 - Use a Pre-Configured Environment]:} Use {.code install_nprophet()} to install Neural Prophet Python Dependencies into a conda environment named {.field nprophet}.")
    cli::cli_li("{.strong [Option 2 - Use a Custom Environment]:} Before running {.code library(neuralprophet)}, use {.code Sys.setenv(NPROPHET_PYTHON = 'path/to/python')} to set the path of your python executable that is located in an environment that has 'numpy', 'pandas', 'pillow', 'torch', 'matplotlib' and 'neuralprophet' available as dependencies.")
    cli::cli_end("nprophet_installation_options")
    cli::cli_h1("End Python Dependency Check")
}

msg_error <- function(e) {
    cli::cli_h1("Error Loading Python Dependencies {.pkg neuralprophet}")
    cli::cli_alert_danger("Python Dependency LoadError")
    cli::cli_text(e)
    cli::cli_h1("End Python Package Load Check")
}




# PACKAGE ENVIRONMENT SETUP ----

pkg.env            <- new.env()
pkg.env$env_name   <- "nprophet"
pkg.env$activated  <- FALSE

# PYTHON DEPENDENCIES ----
# Move Python Imports to Package Environment
# - CRAN comment: Cannot use <<- to modify Global env
pkg.env$nprophet   <- NULL
pkg.env$pd         <- NULL

# ONLOAD ----

.onLoad <- function(libname, pkgname) {

    # ATTEMPT TO CONNECT TO A GLUONTS PYTHON ENV ----
    activate_nprophet()

    # ATTEMPT TO LOAD PYTHON LIBRARIES FROM GLUONTS ENV ----
    dependencies_ok <- check_nprophet_dependencies()
    if (dependencies_ok) {

        try({
            pkg.env$nprophet <- reticulate::import("neuralprophet", delay_load = TRUE, convert = FALSE)
            pkg.env$pd       <- reticulate::import("pandas", delay_load = TRUE, convert = FALSE)
        }, silent = TRUE)

        if (is.null(pkg.env$nprophet)) dependencies_ok <- FALSE
    }

    # LET USER KNOW IF DEPENDENCIES ARE NOT OK ----
    if (!dependencies_ok) {
        if (interactive()) msg_no_nprophet()
    }

    # LOAD MODELS ----
    make_neuralprophet()


}




