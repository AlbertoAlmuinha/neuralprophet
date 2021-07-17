# Split Events----

#' Function to split events
#'
#' @param df Data frame or tibble
#' @param pattern Search pattern

#' @export
split_events <- function(df, pattern = "events"){

    split_df <- list()

    df <- df %>% dplyr::relocate(ds) %>% dplyr::select(ds, dplyr::contains(pattern))

    for(j in 1:(dim(df)[2]-1)){

        split_df[[j]] <- df[, c(1, j+1)]
    }

    for(j in 1:length(split_df)){
        split_df[[j]] <- split_df[[j]][which(split_df[[j]][,2]==1),]
    }

    return(split_df)

}

# Split Convert----

#' Function to Convert events
#'
#' @param split_df Data frame or tibble

#' @export
split_convert <- function(split_df){

    res <- list()

    for(j in 1:length(split_df)){
        res[[j]] <- pkg.env$pd$DataFrame(list(event = names(split_df[[j]])[2], ds = pkg.env$pd$to_datetime(split_df[[j]]$ds)))
    }

    res_df <- pkg.env$pd$concat(reticulate::tuple(res[[1]], res[[2]]))

    if (length(res) > 2){
        for(j in 3:length(res)){
            res_df <- pkg.env$pd$concat(reticulate::tuple(res_df, res[[j]]))
        }
    }

    return(res_df)

}

# Get Combination Value----

#' Function to Get a Number with the Combination Value
#'
#' @param args List with the arguments passed to the function.

#' @export
get_combination_value <- function(args){

    if (any(names(args) == "add_future_regressor") & any(names(args) == "add_events") & any(names(args) == "add_lagged_regressor")) {
        val <- "1"
    }

    if (any(names(args) == "add_future_regressor") & !any(names(args) == "add_events") & any(names(args) == "add_lagged_regressor")) {
        val <- "2"
    }

    if (any(names(args) == "add_future_regressor") & any(names(args) == "add_events") & !any(names(args) == "add_lagged_regressor")) {
        val <- "3"
    }

    if (!any(names(args) == "add_future_regressor") & any(names(args) == "add_events") & any(names(args) == "add_lagged_regressor")) {
        val <- "4"
    }

    if (any(names(args) == "add_future_regressor") & !any(names(args) == "add_events") & !any(names(args) == "add_lagged_regressor")) {
        val <- "5"
    }

    if (!any(names(args) == "add_future_regressor") & any(names(args) == "add_events") & !any(names(args) == "add_lagged_regressor")) {
        val <- "6"
    }

    if (!any(names(args) == "add_future_regressor") & !any(names(args) == "add_events") & any(names(args) == "add_lagged_regressor")) {
        val <- "7"
    }

    if (!any(names(args) == "add_future_regressor") & !any(names(args) == "add_events") & !any(names(args) == "add_lagged_regressor")) {
        val <- "8"
    }

    return(val)

}

# Get Combination Df----

#' Function to Get the Future Df
#'
#' @param new_data Data frame or tibble
#' @param val Value from get_combination_value()
#' @param date_col Date column name
#' @param model Model
#' @param events_df Data frame with the events
#' @param regressors_df Data frame with the regressors

#' @export
get_combination_df <- function(new_data, val, date_col, model, events_df, regressors_df){

    var_future <- names(regressors_df)

    df <- switch(val,
                 "1" = new_data %>% dplyr::select(ds, y, dplyr::all_of(var_future), dplyr::contains("events"), dplyr::contains("lagged")),
                 "2" = new_data %>% dplyr::select(ds, y,  dplyr::all_of(var_future), dplyr::contains("lagged")),
                 "3" = new_data %>% dplyr::select(ds, y,  dplyr::all_of(var_future), dplyr::contains("events")),
                 "4" = new_data %>% dplyr::select(ds, y,  dplyr::contains("events"), dplyr::contains("lagged")),
                 "5" = new_data %>% dplyr::select(ds, y,  dplyr::all_of(var_future)),
                 "6" = new_data %>% dplyr::select(ds, y,  dplyr::contains("events")),
                 "7" = new_data %>% dplyr::select(ds, y,  dplyr::contains("lagged")),
                 "8" = new_data %>% dplyr::select(ds, y)
                 )

    future_df <- switch(val,
                        "1" = model$make_future_dataframe(df, n_historic_predictions = dim(df)[1], events_df = events_df, regressors_df = regressors_df),
                        "2" = model$make_future_dataframe(df, n_historic_predictions = dim(df)[1], events_df = events_df),
                        "3" = model$make_future_dataframe(df, n_historic_predictions = dim(df)[1], events_df = events_df, regressors_df = regressors_df),
                        "4" = model$make_future_dataframe(df, n_historic_predictions = dim(df)[1], events_df = events_df),
                        "5" = model$make_future_dataframe(df, n_historic_predictions = dim(df)[1], regressors_df = regressors_df),
                        "6" = model$make_future_dataframe(df, n_historic_predictions = dim(df)[1], events_df = events_df),
                        "7" = model$make_future_dataframe(df, n_historic_predictions = dim(df)[1]),
                        "8" = model$make_future_dataframe(df, n_historic_predictions = dim(df)[1]))

    return(future_df)

}







