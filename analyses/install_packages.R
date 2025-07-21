# Required R packages
required_packages <- c(
  "tidyr", "dplyr", "readr", "ggplot2", "tinytable", "kableExtra",
  "modelsummary", "fixest", "here", "patchwork", "grid", "scales",
  "data.tree", "DiagrammeR", "vtree", "xtable", "ggnewscale",
  "forcats", "purrr", "gridExtra", "stringr", "rlang", "tinytex",
  "stringr", "psych", "latex2exp", "yaml"
)

# Install missing R packages
installed <- rownames(installed.packages())
missing <- setdiff(required_packages, installed)
if (length(missing)) {
  install.packages(missing)
}

# Ensure tinytex and LaTeX distribution are available
if (!"tinytex" %in% installed) {
  install.packages("tinytex")
}

# Install TeX distribution if not already present
if (!tinytex::is_tinytex()) {
  message("Installing TinyTeX (TeX distribution)...")
  tinytex::install_tinytex()
}
