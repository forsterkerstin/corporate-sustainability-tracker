# corporate-sustainability-tracker
This repository accompanies the paper *"Assessing corporate sustainability with large language models: Evidence from Europe"*.

## Abstract
Companies play a crucial role in achieving global sustainability goals, yet evidence on their progress across environmental, social, and governance (ESG) dimensions remains limited. We develop a machine learning framework to systematically extract ESG indicators from corporate reports. Applying this approach to annual and sustainability reports of 600 large European firms (2014–2023), we construct a dataset of 2.9 million ESG observations across environmental, social, and governance topics. We assess ESG transparency based on disclosures aligned with the European Sustainability Reporting Standards (ESRS) and evaluate ESG performance using extracted numerical indicators. Results reveal a pronounced transparency gap: firms in the top ESG rating decile disclose 22% more indicators than those in the bottom decile, although this gap narrows over time. Performance trends are uneven: while most social indicators remain largely stagnant, except for gains in gender equality, environmental indicators show some improvement. Reported scope 3 emissions increase sharply, largely reflecting improved disclosure. Our open-source framework enables systematic tracking of corporate ESG efforts.

## Installation
### Prerequisites
- **Python 3.8+** (for running the ML pipeline to extract ESG indicators)
- **[Together AI](https://www.together.ai/) API key** (for LLM access during ESG indicator extraction)
- **[Huggingface](https://huggingface.co/) token** (for embedding model access during ESG indicator extraction)
- **R (>= 4.0)** (for data analysis using Quarto)
- **[Quarto CLI](https://quarto.org/docs/get-started/)** (for rendering `.qmd` reports)

### Setup instructions
#### Python environment
Install the required Python packages:

```bash
pip3 install -r requirements.txt
```
#### R environment
Ensure R is installed. Packages are installed automatically upon running the main analysis script `analyses.qmd`.

#### Quarto CLI
Install from [quarto.org](https://quarto.org/docs/get-started/) and verify via:

```bash
quarto --version
```

### Download the dataset

To reproduce the results, download and extract the required data from OSF as follows:

1. Download the ZIP archive from [OSF](https://osf.io/q2jpv/).

2. Unzip the contents into the `data/` folder located in the root directory of the repository. The directory structure should look like this after extraction:
```
data/
├── raw/
│   └── datasets/
│       ├── companies.csv
│       ├── fed_rates_yearly.csv
│       ├── indicator_metadata.csv
│       ├── manual_validation_set_annotator2.csv
│       ├── manual_validation_set.csv
│       └── report_ids.csv
└── processed/
    └── results/
        ├── esg_indicators_postprocessed.csv
        ├── esg_indicators_validation.csv
        └── reports_per_company_year.csv
```
3. Get the corporate report PDF files from the [download link](https://syncandshare.lrz.de/getlink/fiVfpX83ZUsRrKLk2YWenN/) and place them in the folder `data/raw/reports_pdf`. Upon publication, the reports will be available via the [Sustainability Reporting Navigator](https://api.srnav.com/). A corresponding list of report IDs, which can be queried from the endpoint, is provided in `report_ids.csv`.

4. (Optional) Get the proprietary data and place them in the folder `data/raw/datasets`. Exact instructions on file naming are provided in `notebooks/prepare_data.ipynb`.
   - [LSEG - Worldscope Fundamentals](https://www.lseg.com/en/data-analytics/financial-data/company-data/fundamentals-data/worldscope-fundamentals)
   - [LSEG - Refinitiv ESG Scores](https://www.lseg.com/en/data-analytics/sustainable-finance/esg-scores)
   - [MSCI ESG Ratings](https://www.msci.com/data-and-analytics/sustainability-solutions/esg-ratings)

## Workflow overview
The repository implements the following end-to-end workflow from downloaded corporate reports to the released CSV files used in the analyses:

1.  Corporate annual and sustainability reports are downloaded and stored in `data/raw/reports_pdf/`. The file `report_ids.csv` lists the report IDs of the source documents included in the dataset.
2.  The extraction pipeline (`corporate_sustainability_tracker/main.py`) parses the report PDFs and extracts ESG indicators using retrieval-augmented generation.
3.  The extracted outputs are postprocessed and standardized in `notebooks/postprocess_units.ipynb`, including unit harmonization and currency conversion where applicable.
4.  A validation version of the extracted indicators without currency conversion is generated in `notebooks/postprocess_units_for_validation.ipynb`.
5.  Additional input and proprietary datasets are merged in `notebooks/prepare_data.ipynb` to prepare the final analysis data.
6.  The released output files in `data/processed/results/` are then used in the Quarto analysis script (`analyses/analyses.qmd`) to reproduce the reported results.

## CSV file metadata
For reproducibility, we provide brief metadata for the CSV files used in the pipeline and analyses.

### Input and reference files (`data/raw/datasets/`)
- `companies.csv`: Company-level reference file with one row per company-year, including company ID, ISIN, company name, country, primary SICS sector, and year. It is used to link companies to reports and organize the analysis sample.
- `fed_rates_yearly.csv`: Reference file with yearly exchange rates against the U.S. dollar for multiple currencies. It is used to convert reported monetary values into a common currency during postprocessing.
- `indicator_metadata.csv`: Reference file with one row per ESG indicator, including its topic, disclosure requirement, label, context, data type, and related classification fields. It is used to define the indicators extracted and analyzed in the pipeline.
- `manual_validation_set.csv`: Main manual validation file with ESG data points and annotated values. It serves as the primary manual validation set for assessing extraction accuracy.
- `manual_validation_set_annotator2.csv`: Second-annotator version of the manual validation file for the same ESG data points. It is used to assess annotation consistency and inter-annotator agreement.
- `report_ids.csv`: Reference file listing the report IDs of the corporate reports included in the dataset.

### Released output files (`data/processed/results/`)
- `esg_indicators_postprocessed.csv`: Main postprocessed ESG indicator dataset used in the analyses. The unit of observation is an extracted ESG indicator at the company-year level after postprocessing.
- `esg_indicators_validation.csv`: Validation version of the extracted ESG indicators without currency conversion. This file is used for the manual validation exercises.
- `reports_per_company_year.csv`: Report-level coverage information by company and year. This file indicates which reports are available and used for each company-year.

## Usage
This repository consists of three main components:
1. **ESG indicator extraction** using retrieval-augmented generation (RAG) (Python)
2. **Postprocessing and dataset preparation** for the analyses (Jupyter Notebooks)
3. **Analysis and visualization** of corporate ESG transparency and performance (R + Quarto)
   
### Extract ESG indicators
1. Specify your Together AI API key and Huggingface token in the configuration file `configs/config.yaml`.
2. Adjust the remaining parameters as needed.
3. To run the full extraction pipeline, from corporate report parsing to saving of the results: 
```
python corporate_sustainability_tracker/main.py
```
- **Monitoring**: Logs will be generated to track the progress of the pipeline. You can view these logs in the `logs/` directory.
- **Output**: The results will be saved in the `results/` directory, with subdirectories created based on the current timestamp.
- **Error Handling**: If any errors occur, the pipeline is designed to log them and, where possible, continue execution. Review the log files in the `logs/` directory for detailed error messages.

### Postprocess the results and prepare the analysis data
- To postprocess the results (unit parsing and standardization):
  ```
  jupyter notebook notebooks/postprocess_units.ipynb
  ```
- To postprocess the results for validation (i.e., without currency conversion):
  ```
  jupyter notebook notebooks/postprocess_units_for_validation.ipynb
  ```
- To prepare the proprietary analysis data:
  ```
  jupyter notebook notebooks/prepare_data.ipynb
  ```
- To prepare the supplementary analyses:
  ```
  jupyter notebook notebooks/prepare_supplementary_analyses.ipynb
  ```

### Reproduce the analysis
The analysis is conducted in R using [Quarto](https://quarto.org). The core script `analyses.qmd` compiles the main results. Additional `.qmd` modules in the `chunks/` folder contain supplementary and figure-specific code.
To reproduce the main analysis on root level:

```bash
quarto render analyses/analyses.qmd
```

## GUIDE-LLM reporting checklist

This repository includes the completed [GUIDE-LLM reporting checklist](https://sfeuerriegel.github.io/llm-checklist/) for the study:
[supplementary_materials/GUIDE-LLM_checklist.pdf](supplementary_materials/GUIDE-LLM_checklist.pdf)
