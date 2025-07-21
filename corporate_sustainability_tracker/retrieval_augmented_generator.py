import concurrent.futures
import json
import logging
import os
import re
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import yaml
from huggingface_hub import login
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_together import Together

# from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer
from vectorstore_manager import VectorstoreManager


class RetrievalAugmentedGenerator:
    def __init__(
        self,
        config: Dict[str, Any],
        log_filename: str,
        vectorstore_manager: VectorstoreManager,
    ) -> None:
        """
        Initializes the `RetrievalAugmentedGenerator` class with the given
        configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
            vectorstore_manager (VectorstoreManager): An instance of the
            `VectorstoreManager´ class.
        """
        self._config_file = config
        self._log_filename = log_filename
        self._huggingface_token = config["huggingface_token"]
        self._text_file_type = config["text_file_type"]
        self._srn_api_link = config["srn_api_link"]
        # Initialize file paths and directories
        self._existing_results_folder = config["existing_results_folder"]
        self._data_points_file = config["data_points_file"]
        self._results_dir = config["results_dir"]
        # Initialize maximum number of workers for ThreadPoolExecutor
        self._max_workers = config["max_workers"]
        # Initialize Together API key
        self._together_api_key = config["together_api_key"]
        # Initialize model parameters
        self._inference_model_name = config["inference_model_name"]
        self._model_temperature = config["model_temperature"]
        self._model_output_max_tokens = config["model_output_max_tokens"]
        self._system_prompt = config["system_prompt"]
        self._user_prompt = config["user_prompt"]
        self._inference_model = None
        # Initialize maximum number of retries for generation
        self._max_generation_retries = config["max_generation_retries"]
        # Initialize vectorstore manager
        self._vectorstore_manager = vectorstore_manager
        # Initialize reranking parameters
        self._apply_reranking = config["apply_reranking"]
        self._reranking_model_name = config["reranking_model_name"]
        self._reranking_model = None

    @staticmethod
    def _clean_json_string(json_string: str) -> str:
        """
        Removes commas within numeric values in a JSON-formatted string to
        ensure valid JSON parsing. This is useful for handling cases where
        numbers are incorrectly formatted with commas (e.g., "1,234,567"
        instead of "1234567").

        Args:
            json_string (str): The JSON string with potential comma-separated
            numbers.

        Returns:
            str: The cleaned JSON string with commas removed from numeric
            values.

        Example:
            clean_json_string('{"value": "1,234,567", "unit": "kg"}')
            -> '{"value": "1234567", "unit": "kg"}'
        """
        # Remove commas within numbers (e.g., "2,345,567" -> "2345567")
        json_string = re.sub(r"(?<=\d),(?=\d)", "", json_string)
        return json_string

    def extract_datapoints_concurrently(self) -> str:
        """
        Extracts data points concurrently from documents using Retrieval-
        Augmented Generation (RAG).

        This method identifies all vectorstore files, concurrently processes
        each document and extracts data points using RAG.
        The results are added to a pandas DataFrame, which is saved as a CSV
        file in a dedicated directory.

        Returns:
            str: A string representing the path to the directory where the
                results are saved.

        """
        logging.info("Starting data point extraction process.")
        # Log into huggingface for token count
        login(token=self._huggingface_token)
        # Measure the total runtime for datapoint extraction
        start_time = time.time()

        # Set Together API key as environment variable
        os.environ["TOGETHER_API_KEY"] = self._together_api_key

        # Load data points DataFrame from data_points_dir
        try:
            df_data_points_complete = pd.read_csv(
                self._data_points_file
            )
        except Exception as e:
            logging.error(f"Error loading data points file: {e}.")
            return None

        self._df_data_points = df_data_points_complete

        # Create list of document names to be processed
        document_name_list = [
            f.rstrip(".vectorstore")
            for f in os.listdir(
                self._vectorstore_manager.vectorstore_dir
                + "_"
                + self._text_file_type
            )
            if f.endswith(".vectorstore")
        ]

        # Create company id-name dict
        company_df = pd.read_json(self._srn_api_link + "companies")
        company_dict = (
            company_df[["id", "name"]].set_index("id")["name"].to_dict()
        )

        # Create results dataframe and directories
        self._create_results_df_and_dirs()

        try:
            self._inference_model = Together(
                model=self._inference_model_name,
                temperature=self._model_temperature,
                max_tokens=self._model_output_max_tokens,
            )

        except Exception as e:
            logging.error(f"Error initializing inference model: {e}.")

        # Concurrently run the RAG pipeline using ThreadPoolExecutor
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self._max_workers
        ) as executor:
            try:
                futures = [
                    executor.submit(
                        self._extract_datapoints_from_document,
                        document_name,
                        company_dict,
                    )
                    for document_name in document_name_list
                ]
                results = [
                    future.result()
                    for future in concurrent.futures.as_completed(futures)
                ]
            except Exception as e:
                logging.warning(
                    f"Unhandled exception when retrieving result: {e}"
                )
            except KeyboardInterrupt:
                logging.warning(
                    "KeyboardInterrupt detected, shutting down executor."
                )
                executor.shutdown(cancel_futures=True)
        # Ensure all futures complete
        concurrent.futures.wait(futures)
        num_results = len([x for x in results if x is not None])

        logging.info(
            f"Completed data point extraction for {num_results}/{len(results)}"
            f" documents. Total runtime: {round((time.time()-start_time), 2)}s"
        )
        return self._results_subdir

    def _extract_datapoints_from_document(
        self,
        document_name: str,
        company_dict: Dict[str, str],
    ) -> Optional[List[Dict[str, Optional[str]]]]:
        """
        Extracts data points from a specific document using Retrieval-Augmented
        Generation (RAG).

        This method loads the vectorstore for the specified document,
        retrieves relevant contexts for each data point, and generates outputs
        using a language model.
        The results, including the retrieved context, model output, and other
        relevant information, are returned as a list of dictionaries.

        Args:
            document_name (str): The name of the document to process.

        Returns:
            Optional[List[Dict[str, Optional[str]]]]: A list of dictionaries
            containing extracted data points and their associated information.
            Returns `None` if the vector store could not be loaded.
        """
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.FileHandler(self._log_filename, mode="a")],
        )
        logger = logging.getLogger()
        # Create results file name
        results_file = os.path.join(
            self._results_subdir, document_name + ".parquet.gzip"
        )
        # Extract company_id and year from document_name
        company_id, year = document_name.split("_")
        # Initialize list of results for the current document
        results_list_current_document = []

        # Measure the runtime for each document
        start_time = time.time()

        df_document_data_points_all = self._df_data_points

        # If parquet exists, check for invalid outputs in dataframe
        if os.path.exists(results_file):
            try:
                previous_results_df = pd.read_parquet(
                    results_file, engine="fastparquet"
                )
                df_document_data_points = df_document_data_points_all[
                    (
                        df_document_data_points_all["id"].apply(
                            lambda x: (
                                not previous_results_df.loc[
                                    previous_results_df["data_point_id"] == x,
                                    "model_output_valid",
                                ].values[0]
                                if x
                                in previous_results_df["data_point_id"].values
                                else False
                            )
                        )
                    )
                    | (
                        ~df_document_data_points_all["id"].isin(
                            previous_results_df["data_point_id"]
                        )
                    )
                ]
                if len(df_document_data_points) == 0:
                    logger.info(f"No invalid outputs for {document_name}.")
                    return
            except Exception as e:
                logger.warning(
                    f"Exception filtering existing results: {e}"
                    f"Skipping {document_name}."
                )
                return
        else:
            previous_results_df = None
            df_document_data_points = df_document_data_points_all
        # Load vectorstore of the current document as retriever
        retriever = self._vectorstore_manager.load_vectorstore_as_retriever(
            f"{document_name}.vectorstore"
        )
        if retriever is None:
            logger.warning(
                f"Retriever could not be loaded for document "
                f"{document_name}. Skipping document."
            )
            return None

        for _, row in df_document_data_points.iterrows():

            try:
                # Generate data point description
                data_point_description = (
                    f"{row["label"]} {row["specification"]}"
                    if pd.notna(row["specification"])
                    else row["label"]
                )

                # Measure time to retrieve the context
                start_retrieval = time.time()

                # Retrieve the context for the current data point
                docs = self._vectorstore_manager.retrieve_context_with_retries(
                    retriever,
                    document_name,
                    data_point_description,
                    self._reranking_model,
                )
                if docs is None:
                    logger.warning(
                        f"Context could not be retrieved for "
                        f"{document_name}. Skipping document."
                    )
                    return None

                retrieval_time = time.time() - start_retrieval

                context_list, source_document_list, page_numbers_list = (
                    self._clean_context(docs)
                )
                # Add a chunk identifier and concatenate list elements
                formatted_chunks = []
                for i, chunk in enumerate(context_list):
                    # Short chunk identifier
                    formatted_chunks.append(f"Chunk {i+1}:\n{chunk}")
                # Join all chunks with a newline separator
                context = (
                    "\n\n".join(formatted_chunks)
                    .replace("{", "{{")
                    .replace("}", "}}")
                )

                # Create prompt using system and user prompt
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", self._system_prompt),
                        (
                            "human",
                            self._user_prompt.format(
                                company=company_dict[company_id],
                                year=year,
                                context=context,
                            ),
                        ),
                    ]
                )

                # Count tokens in prompt
                # Initialize your Hugging Face tokenizer
                model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                # Render the full prompt as a string
                rendered_prompt = prompt.format(request=data_point_description)
                # Tokenize the rendered prompt and count the tokens
                tokens = tokenizer(rendered_prompt, return_tensors="pt")
                num_tokens = len(tokens["input_ids"][0])

                # Define the LLM chain
                model_chain = (
                    {"request": RunnablePassthrough()}
                    | prompt
                    | self._inference_model
                    | StrOutputParser()
                )
                model_output = None
                model_output_valid = False
                value = None
                unit = None

                # Measure time to extract data point
                start_extraction = time.time()

                # Invoke the LLM chain
                model_output = self._invoke_chain_with_retries(
                    document_name,
                    model_chain,
                    data_point_description,
                    logger,
                    max_retries=self._max_generation_retries,
                )
                extraction_time = time.time() - start_extraction

                if model_output is not None:
                    if set(model_output.keys()) == {"value", "unit"}:
                        model_output_valid = True
                        value = model_output["value"]
                        unit = model_output["unit"]

            except Exception as e:
                traceback_str = "".join(traceback.format_tb(e.__traceback__))
                logger.warning(
                    f"{company_id}_{year} "
                    f"{data_point_description}: Exception during "
                    f"datapoint extraction: {e}."
                    f"Traceback: {traceback_str}."
                    f"Flagging invalid ouptput."
                )
                model_output = None
                model_output_valid = False
                value = None
                unit = None
                context_list = None
                source_document_list = None
                page_numbers_list = None
                num_tokens = None

            # Append the document information, context, model output and
            # extraction time to the results_list of the current document
            results_list_current_document.append(
                {
                    "company_id": company_id,
                    "year": year,
                    "company_year": document_name,
                    "data_point_id": row["id"],
                    "data_point_description": data_point_description,
                    "srn_compliance_item_id": row["srn_compliance_item_id"],
                    "model_output": model_output,
                    "model_output_valid": model_output_valid,
                    "value": str(value),
                    "unit": unit,
                    "retrieved_context": context_list,
                    "source_documents": source_document_list,
                    "page_numbers": page_numbers_list,
                    "retrieval_time": retrieval_time,
                    "extraction_time": extraction_time,
                    "num_tokens": num_tokens,
                }
            )

        # Log runtime for processing the current document
        logger.info(
            f"Data point extraction complete for document {document_name}."
            f"Runtime: {round((time.time()-start_time), 2)} s"
        )
        # Ensure memory gets freed
        del retriever
        torch.cuda.empty_cache()

        # Update the .csv file
        try:
            # Add the results list of the current document to the results file
            results_df = pd.DataFrame(results_list_current_document)
            if previous_results_df is not None:
                results_df = pd.concat(
                    [previous_results_df, results_df], ignore_index=True
                )
                results_df = results_df.sort_values(
                    by="model_output_valid", ascending=False
                )
                results_df = results_df.drop_duplicates(
                    subset=["data_point_id", "company_year"], keep="first"
                )
            # Save to parquet
            results_df.to_parquet(
                results_file,
                compression="gzip",
                index=False,
                engine="fastparquet",
            )
        except Exception as e:
            logger.warning(
                f"Exception during csv file writing for {document_name}: {e}."
            )
        return 0

    def _invoke_chain_with_retries(
        self,
        document_name: str,
        chain: RunnableSequence,
        input: str,
        logger: logging.Logger,
        max_retries: int = 10,
    ) -> Optional[str]:
        """
        Invokes a LangChain model chain with retry logic.

        This method attempts to invoke a specified model chain with the
        provided input. If the invocation fails, it retries up to `max_retries`
        times, waiting 5 seconds between each attempt. If all retries fail, it
        logs a warning and returns `None`.

        Args:
            document_name (str): The name of the document being processed.
            chain (RunnableSequence): The LangChain model chain to be invoked.
            input (str): The input string to be passed to the model chain.
            max_retries (int, optional): The maximum number of retry attempts.
            Defaults to 10.

        Returns:
            Optional[str]: The output string from the model chain if
            successful, or `None` if all retries fail.
        """
        retry_attempts = 0
        while retry_attempts < max_retries:
            try:
                # Invoke the model chain and return the output
                raw_output = chain.invoke(input)
                cleaned_output = self._clean_json_string(raw_output)
                model_output = json.loads(cleaned_output)
                return model_output
            except Exception as e:
                logger.warning(f"Exception during model invoke: {e}. Retry.")
                retry_attempts += 1
                time.sleep(3)

        # If all retries failed, log and return None
        logger.warning(
            f"All model invoke retries failed for data point `{input}` in"
            f" document {document_name}."
        )
        return None

    def _create_results_df_and_dirs(self) -> None:
        """
        Creates or loads a DataFrame for storing results and sets up the
        necessary directories and file paths.

        If an existing results file is found at `self._existing_results_file`,
        it reads the file into a DataFrame, sets the results directory and
        file paths accordingly, and logs the action. If no such file exists,
        it initializes an empty DataFrame with predefined columns, creates a
        new subdirectory for storing results based on the current timestamp,
        and sets up the file paths for the new results file.
        """
        # If an existing_results_folder exists, read as DataFrame
        if os.path.exists(self._existing_results_folder):
            logging.info(
                f"Adding to existing results: {self._existing_results_folder}."
            )
            self._results_subdir = self._existing_results_folder
        else:
            # If no existing_results file exists, initialize an empty DataFrame
            logging.info("No existing results found. Creating a new file.")
            # Create a subdirectory for the results using the current timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._results_subdir = os.path.join(
                self._results_dir, timestamp, "raw_results"
            )
            os.makedirs(self._results_subdir, exist_ok=True)

        # Save config to results directory
        config_path = os.path.join(
            os.path.dirname(self._results_subdir), "config.yaml"
        )

        # Save the config as YAML
        with open(config_path, "w") as file:
            yaml.safe_dump(self._config_file, file)

        logging.info(
            f"Saving the extracted data points in {self._results_subdir}."
        )

    def _clean_context(self, docs):
        """
        Processes retrieved documents to separate text content, source
        documents, and page numbers.

        Parameters:
        - docs (List[Document]): A list of LangChain Document objects
        containing text and metadata.

        Returns:
        - context (str): Formatted string of text content with metadata for
        each document.
        - source_documents (List[str]): List of source document names.
        - page_numbers (List[int]): List of page numbers for each document.
        """
        context_list = []
        source_document_list = []
        page_numbers_list = []

        for doc in docs:
            # Extract page content and metadata from each document
            text = doc.page_content
            # Append formatted text and metadata for context
            context_list.append(text)

            if self._text_file_type == "json":
                try:
                    source_document = doc.metadata["source_info"][0].get(
                        "source_document", "Unknown"
                    )
                    page_numbers = [
                        info.get("page_number", None)
                        for info in doc.metadata["source_info"]
                    ]
                    source_document_list.append(source_document)
                    page_numbers_list.append(page_numbers)
                except Exception:
                    source_document_list = ["Unknown"]
                    page_numbers_list = [None]

        return context_list, source_document_list, page_numbers_list
