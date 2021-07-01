library(testthat)
library(timetk)
library(tidymodels)
library(modeltime)
library(tidyverse)
library(neuralprophet)


skip_if_no_nprophet <- function() {

    nprophet_available <- FALSE

    try({
        nprophet_available <- reticulate::py_module_available("neuralprophet")
    }, silent = TRUE)

    if (!nprophet_available) {
        skip("neuralprophet not available for testing")
    }
}


test_check("neuralprophet")
