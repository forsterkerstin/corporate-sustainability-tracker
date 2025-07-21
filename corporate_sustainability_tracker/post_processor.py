import glob
import logging
import os

import pandas as pd


class PostProcessor:
    def __init__(
        self,
        results_subdir: str,
    ) -> None:
        """
        Initializes the `PostProcessor` class with the given configuration,
        raw results DataFrame, and the directory where results
        will be stored.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
            raw_results_df (pd.DataFrame): DataFrame containing the raw
            results.
            results_subdir (str): Path to the directory where the processed
            results will be saved.
        """
        self._results_subdir = os.path.dirname(results_subdir)

    def process_and_save_results(self) -> None:
        """
        Processes the model output by extracting numerical values and
        saves the results in the specified directory.
        """
        logging.info(
            f"Starting postprocessing. Saving results in "
            f"{self._results_subdir}."
        )

        # Use glob to find all .parquet files in the directory
        parquet_dir = os.path.join(self._results_subdir, "raw_results")
        parquet_files = glob.glob(f"{parquet_dir}/*.parquet.gzip")

        # Initialize an empty list to hold DataFrames
        dataframes = []

        # Loop over all parquet files and read them with gzip compression
        for file in parquet_files:
            try:
                # Append each DataFrame to the list
                dataframes.append(pd.read_parquet(file, engine="fastparquet"))
            except Exception as e:
                logging.warning(
                    f"Exception appending results for file {file}: "
                    f"{e}. Skipping."
                )
                continue

        # Concatenate all DataFrames in the list into a single DataFrame
        self._final_results_df = pd.concat(dataframes, ignore_index=True)

        # Save as .csv
        file_complete = os.path.join(
            self._results_subdir, "raw_results_complete.csv"
        )
        self._final_results_df.to_csv(file_complete, index=False)

        self._final_results_df.drop(
            [
                "retrieved_context",
                "source_documents",
                "page_numbers",
                "retrieval_time",
                "extraction_time",
                "num_tokens",
            ],
            axis="columns",
            inplace=True,
        )

        if self._final_results_df.shape[0] == 0:
            logging.warning("No results to process.")
            return None

        # Save the numerical dataframe as CSV file
        self._save_numerical_results()

        logging.info("Completed postprocessing!")

    def _save_numerical_results(self) -> None:
        """Save the numerical results DataFrame as a CSV file."""
        file_numerical = os.path.join(
            self._results_subdir, "numerical_results.csv"
        )
        self._final_results_df.drop("model_output", axis="columns").to_csv(
            file_numerical, index=False
        )
