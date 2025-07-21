import concurrent.futures
import json
import logging
import os
import time
import traceback
from typing import Any, Dict, List, Tuple

import fitz
import pymupdf4llm
from cleantext import clean


class TextExtractor:
    def __init__(self, config: Dict[str, Any], log_filename) -> None:
        """
        Initializes the `TextExtractor` class with the given configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
        """
        # Initialize class attributes from the configuration file
        self._log_filename = log_filename
        self._pdf_dir = config["pdf_dir"]
        self._txt_dir = config["txt_dir"]
        self._text_file_type = config["text_file_type"]
        self._max_workers = config["max_workers"]

    def extract_and_save_text_concurrently(self) -> None:
        """
        Groups the PDF files by company and year.
        Extracts and concatenates the text.
        Saves the text as TXT/MD files in the dedicated directory.
        """
        logging.info("Starting text extraction and saving process.")
        # Measure the runtime
        start_time = time.time()
        # Create the directory for the TXT files if it does not exist
        os.makedirs(self._txt_dir, exist_ok=True)
        # Group documents by company_year
        documents_grouped = self._group_documents_by_company_year()

        # For each company_year group, extract, concatenate and save text
        tasks = [
            (company_id, year, pdf_names)
            for (company_id, year), pdf_names in documents_grouped.items()
        ]
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self._max_workers
        ) as executor:
            try:
                futures = [
                    executor.submit(
                        self._extract_and_save_text_for_company_year, *task
                    )
                    for task in tasks
                ]
            except KeyboardInterrupt:
                logging.warning(
                    "KeyboardInterrupt detected, shutting down executor."
                )
                executor.shutdown(cancel_futures=True)
        # Ensure all futures complete
        concurrent.futures.wait(futures)

        # Collect results from all futures
        pdfs_with_error = [
            future.result()
            for future in futures
            if future.result() is not None
        ]

        logging.warning(
            f"{len(pdfs_with_error)} PDF documents with text extraction error:"
            f" {pdfs_with_error}"
        )
        logging.info(
            f"Completed text extraction and saving process. Runtime: "
            f"{round((time.time()-start_time), 2)} s"
        )

    def _extract_and_save_text_for_company_year(
        self,
        company_id: str,
        year: str,
        pdf_names: List[str],
    ) -> None:
        """
        Extracts text from the PDF files for a given company and year,
        and saves it as a TXT file.

        Args:
            company_id (str): The ID of the company.
            year (str): The year associated with the documents.
            pdf_names (List[str]): List of PDF file names.
        """
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.FileHandler(self._log_filename, mode="a")],
        )
        logger = logging.getLogger()
        # Create file path for the TXT/MD file
        file_name = f"{company_id}_{year}.{self._text_file_type}"
        file_path = os.path.join(self._txt_dir, file_name)

        # Check if output file for company_year alreay exists
        if os.path.exists(file_path):
            logger.info(
                f"Output file for {company_id}, {year} already exists "
                f"and saved as {file_name}. Skipping."
            )
            return None

        # combined_text = []
        # New approach to keep metadata
        all_text = ""
        metadata = []
        for pdf_name in pdf_names:
            try:
                # Extract text/markdown from PDF file with PyMuPDF(4LLM)
                if self._text_file_type == "md":
                    markdown_doc = self._extract_markdown_with_pymupdf(
                        pdf_name
                    )
                    all_text += markdown_doc
                elif self._text_file_type == "json":
                    pages = self._extract_text_with_metadata(pdf_name)
                    for page in pages:
                        all_text += page["text"] + " "
                        metadata.append(
                            {
                                "source_document": page["source_document"],
                                "page_number": page["page_number"],
                                "text_length": len(page["text"]),
                            }
                        )
                else:
                    logger.error(
                        f"Unknown text file type specified "
                        f"[`json`/`md`]: {self._text_file_type}."
                    )
                    return None
            except Exception as e:
                # If the extraction fails, log and skip document
                traceback_str = "".join(traceback.format_tb(e.__traceback__))
                logger.warning(
                    f"Problem with {pdf_name}: {e}. {traceback_str}. Skipping "
                    f"company {company_id} and year {year}."
                )
                return pdf_name
            # Add double newline at each document end
            all_text += "\n\n"
        # Clean text
        cleaned_text = clean(
            # combined_text,
            all_text,
            fix_unicode=True,
            to_ascii=False,
            lower=False,
            no_line_breaks=False,
            strip_lines=False,
            keep_two_line_breaks=True,
            normalize_whitespace=True,
            no_emoji=True,
        )

        # Save text/markdown to file
        if self._text_file_type == "md":
            with open(file_path, "wb") as file:
                file.write(cleaned_text.encode("utf-8"))
        elif self._text_file_type == "json":
            # Save intermediate document with concatenated text and metadata
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"text": all_text, "metadata": metadata},
                    f,
                    ensure_ascii=False,
                )
        logger.info(
            f"Extracted text for {company_id}, {year}, saved in {file_name}."
        )

    def _extract_text_with_metadata(self, pdf_name):
        pdf_path = os.path.join(self._pdf_dir, pdf_name)
        pages = []
        try:
            with fitz.open(pdf_path) as doc:
                for page_num in range(doc.page_count):
                    page_text = doc[page_num].get_text()
                    pages.append(
                        {
                            "text": page_text,
                            "page_number": page_num + 1,
                            "source_document": os.path.basename(pdf_path),
                        }
                    )
        except Exception as e:
            logging.warning(
                f"Failed to extract text from file {pdf_name}. "
                f"Error: {e}. Returning empty string."
            )

        return pages

    def _extract_text_with_pymupdf(self, pdf_name: str) -> str:
        """
        Extracts text from a PDF document using PyMuPDF.

        Args:
            pdf_name (str): The name of the PDF file.

        Returns:
            str: The extracted text from the PDF file.
        """
        pdf_path = os.path.join(self._pdf_dir, pdf_name)
        full_text = []

        # Open the PDF file using a context manager
        with fitz.open(pdf_path) as document:
            # Iterate through each page and extract text
            for page in document:
                full_text.append(page.get_text())

        return " ".join(full_text)

    def _extract_markdown_with_pymupdf(self, pdf_name: str) -> str:
        """
        Extracts markdown text from a PDF document using PyMuPDF(4LLM).

        Args:
            pdf_name (str): The name of the PDF file.

        Returns:
            str: The extracted markdown text from the PDF file.
        """
        pdf_path = os.path.join(self._pdf_dir, pdf_name)
        # Extract markdown text from the PDF file
        markdown_text = pymupdf4llm.to_markdown(
            pdf_path, table_strategy="lines"
        )
        return markdown_text

    def _group_documents_by_company_year(
        self,
    ) -> Dict[Tuple[str, str], List[str]]:
        """
        Groups PDF documents by company_id and year.

        Returns:
            Dict[Tuple[str, str], List[str]]:
            A dictionary with keys as (company_id, year)
            and values as lists of PDF file names.
        """
        documents_grouped = {}
        for filename in os.listdir(self._pdf_dir):
            if filename.endswith(".pdf"):
                company_id, year, _ = filename[:-4].split("_")
                key = (company_id, year)
                if key not in documents_grouped:
                    documents_grouped[key] = []
                documents_grouped[key].append(filename)
        return documents_grouped
