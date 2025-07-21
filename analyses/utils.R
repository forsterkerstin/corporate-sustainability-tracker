add_firm_data <- function(data) {
    companies <- read_csv(file.path(datasets_dir, "companies.csv"), show_col_types = FALSE)

    join_keys <- c("firm", "year")
    overlapping <- intersect(setdiff(colnames(data), join_keys), colnames(companies))
    companies_clean <- companies %>% select(-all_of(overlapping))

    output <- data %>%
        left_join(companies_clean, by = join_keys)

    return(output)
}

add_proprietary_firm_data <- function(data) {
    firm_data <- read_csv(file.path(datasets_dir, "firm_data.csv"), show_col_types = FALSE) %>%
        rename("firm" = "id")

    join_keys <- c("firm", "year")
    overlapping <- intersect(setdiff(colnames(data), join_keys), colnames(firm_data))
    firm_data_clean <- firm_data %>% select(-all_of(overlapping))

    output <- data %>%
        left_join(firm_data_clean, by = join_keys) %>%
        mutate(
            # Factorize ESG scores
            across(
                c(ref_esg_score, ref_esg_combined_score, ref_env_score, ref_soc_score, ref_gov_score),
                ~ factor(.x, levels = c("A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-"))
            ),
            msci_company_rating = factor(
                msci_company_rating,
                levels = c("AAA", "AA", "A", "BBB", "BB", "B", "CCC")
            ),
            # Make scores numeric
            ref_esg_score_n = recode(
                ref_esg_score,
                "A+" = 11, "A" = 10, "A-" = 9,
                "B+" = 8,  "B" = 7,  "B-" = 6,
                "C+" = 5,  "C" = 4,  "C-" = 3,
                "D+" = 2,  "D" = 1,  "D-" = 0
            ),
            msci_company_rating_n = recode(
                msci_company_rating,
                "AAA" = 6, "AA" = 5, "A" = 4,
                "BBB" = 3, "BB" = 2, "B" = 1, "CCC" = 0
            )
        )

    return(output)
}


add_all_firm_data <- function(df) {
    df <- add_firm_data(df)
    if (file.exists(file.path(datasets_dir, "firm_data.csv"))) {
        df <- add_proprietary_firm_data(df)
    }
    return(df)
}
