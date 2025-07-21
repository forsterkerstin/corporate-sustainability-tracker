import concurrent.futures
import json
import logging
import os
import time
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

# from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_together import TogetherEmbeddings
from sentence_transformers import CrossEncoder
from together import Together


class VectorstoreManager:
    def __init__(self, config: Dict[str, Any], log_filename) -> None:
        """
        Initializes the `VectorstoreManager` class with the given
        configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
        """
        self._log_filename = log_filename
        # Initialize directories for text and vectorstore files
        self.txt_dir = config["txt_dir"]
        self.vectorstore_dir = config["vectorstore_dir"]
        # Initialize text file type
        self._text_file_type = config["text_file_type"]
        # Initialize chunking parameters size and overlap
        self._chunk_size = config["chunk_size"]
        self._chunk_overlap = config["chunk_overlap"]
        self._chunk_separators = config["chunk_separators"]
        # Initialize the embedding model
        self._together_api_key = config["together_api_key"]
        self._embedding_model_host = config["embedding_model_host"]
        self._embedding_model_name = config["embedding_model_name"]
        # Initialize maximum number of workers for concurrent processing
        self._max_workers = config["max_workers"]
        # Initialize retrieval parameters
        self._max_retrieval_retries = config["max_retrieval_retries"]
        self._parent_document_retrieval = config["parent_document_retrieval"]
        self._parent_document_window_size = config[
            "parent_document_window_size"
        ]
        self._apply_hybrid_search = config["apply_hybrid_search"]
        self._apply_reranking = config["apply_reranking"]
        self._number_of_chunks = config["number_of_chunks"]
        self._number_of_chunks_reranked = config["number_of_chunks_reranked"]

    def create_vectorstores_concurrently(self) -> None:
        """
        Creates vectorstores concurrently for each text file in the specified
        directory using `ProcessPoolExecutor` with up to `self._max_workers`
        threads.

        Saves the generated vectorstores to `self._vectorstore_dir`.
        """
        logging.info(
            "Starting vectorstore creation process. "
            f"Using embedding model {self._embedding_model_name} "
            f"hosted by {self._embedding_model_host}."
        )
        # Measure the total runtime for vectorstore creation
        start_time = time.time()

        # Create a list of all company_years for which to create vectorstores
        company_year_list = [
            file.removesuffix(f".{self._text_file_type}")
            for file in os.listdir(self.txt_dir)
            if file.endswith(f".{self._text_file_type}")
        ]

        # Create vectorstores for all company_years using ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self._max_workers
        ) as executor:
            try:
                futures = [
                    executor.submit(
                        self._create_and_save_vectorstore, company_year
                    )
                    for company_year in company_year_list
                ]
                results = [
                    future.result()
                    for future in concurrent.futures.as_completed(futures)
                ]
            except KeyboardInterrupt:
                logging.warning(
                    "KeyboardInterrupt detected, shutting down executor."
                )
                executor.shutdown(cancel_futures=True)
        # Ensure all futures complete
        concurrent.futures.wait(futures)

        logging.info(
            f"Completed vectorstore creation process for "
            f"{len([x for x in results if x is not None])}/{len(results)} "
            f"company-years. Total runtime: "
            f"{round((time.time()-start_time), 2)} s"
        )

    def load_vectorstore_as_retriever(
        self, vectorstore_name: str
    ) -> Optional[VectorStoreRetriever]:
        """
        Loads a FAISS-based vectorstore from a file and initializes it as a
        retriever.
        Configures the retriever to extract the specified number of chunks
        `self._number_of_chunks` during retrieval.
        Defaults to cosine similarity search.

        Args:
            vectorstore_name (str): The name of the vectorstore file to load.

        Returns:
            Optional[VectorStoreRetriever]: The initialized retriever, or
            `None` if loading fails.
        """

        if self._embedding_model_host == "TogetherAI":
            httpx_logger = logging.getLogger("httpx")
            httpx_logger.setLevel(logging.WARNING)
            # Set Together API key as environment variable
            os.environ["TOGETHER_API_KEY"] = self._together_api_key
            embedding_model = TogetherEmbeddings(
                model=self._embedding_model_name,
            )
        elif self._embedding_model_host == "HuggingFace":
            embedding_model = HuggingFaceEmbeddings(
                model_name=self._embedding_model_name
            )
        else:
            logging.warning(
                "Invalid embedding model host specified. Returning None."
            )
            return None

        vectorstore_file = os.path.join(
            self.vectorstore_dir + "_" + self._text_file_type, vectorstore_name
        )

        try:
            vectorstore = FAISS.load_local(
                vectorstore_file,
                embedding_model,
                allow_dangerous_deserialization=True,
            )
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": self._number_of_chunks}
            )
            return retriever
        except Exception as e:
            logging.warning(
                f"Vectorstore {vectorstore_name} cannot be loaded as retriever"
                f" due to the following error: {e}. Skipping document."
            )
            return None

    def retrieve_context_with_retries(
        self,
        retriever: VectorStoreRetriever,
        document_name: str,
        data_point_description: str,
        reranking_model: CrossEncoder,
    ) -> Optional[List[str]]:
        """
        Retrieves relevant context for a given `data_point_description` from
        the provided document invoking the provided retriever with multiple
        retry attempts (`self._max_retrieval_retries`).
        If all attempts fail, logs a warning and returns `None`.

        Args:
            retriever (VectorStoreRetriever): The VectorStoreRetriever object
            used to invoke the retrieval process.
            document_name (str): The name of the document being processed.
            data_point_description (str): The description of the data point
            for which context is being retrieved.

        Returns:
            Optional[List[str]]: The cleaned context list if retrieval is
            successful; otherwise, `None` if all retries fail.
        """
        # If hybrid search is specified, initialize keyword_retriever
        if self._apply_hybrid_search:
            chunks = [
                item[1]
                for item in retriever.vectorstore.docstore._dict.items()
            ]
            keyword_retriever = BM25Retriever.from_documents(chunks)
            keyword_retriever.k = self._number_of_chunks
            retriever = EnsembleRetriever(
                retrievers=[retriever, keyword_retriever], weights=[0.5, 0.5]
            )
        retry_attempts = 0
        while retry_attempts < self._max_retrieval_retries:
            try:
                docs = retriever.invoke(data_point_description)
                if self._apply_reranking:
                    """
                    context = self._rerank_context(
                        reranking_model,
                        data_point_description,
                        # self._clean_context(docs),
                        docs,
                        top_k=self._number_of_chunks_reranked,
                    )
                    """
                    response = Together().rerank.create(
                        model="Salesforce/Llama-Rank-V1",
                        query=data_point_description,
                        documents=[doc.page_content for doc in docs],
                        top_n=self._number_of_chunks_reranked,
                    )
                    context = [
                        docs[result.index] for result in response.results
                    ]
                else:
                    context = docs
                # If parent document retriever is specified, retrieve
                # neighboring chunks
                if self._parent_document_retrieval is True:
                    concatenated_context = self._retrieve_parent_documents(
                        retriever,
                        context,
                        document_name,
                        data_point_description,
                    )
                    # Return all parent docs with original metadata
                    return concatenated_context
                return context
            except Exception as e:
                traceback_str = "".join(traceback.format_tb(e.__traceback__))
                logging.warning(
                    f"Exception during retrieval: {e}"
                    f"Traceback: {traceback_str}"
                )
                retry_attempts += 1
                time.sleep(5)
        if retry_attempts == self._max_retrieval_retries:
            logging.warning(
                "Cannot invoke retriever for data point "
                f"{data_point_description} in document {document_name}."
                "Skipping document."
            )
            return None

    def _create_and_save_vectorstore(self, company_year: str) -> Optional[int]:
        """
        Creates and saves a vectorstore for a given company year.

        This method generates a vectorstore from a text file corresponding to
        the specified `company_year`. If the vectorstore does not exist yet,
        it divides the text into chunks, creates the vector store using the
        FAISS library, and saves it to a dedicated directory. If any errors
        occur during the process, they are logged, and the method returns
        `None`.

        Args:
            company_year (str): The company_year ID for the document being
            processed.

        Returns:
            Optional[int]: Returns `0` if the vector store is successfully
            created and saved. Returns `None` if the vector store already
            exists or if an error occurs.
        """
        # Setup logging for multiprocessing
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.FileHandler(self._log_filename, mode="a")],
        )
        logger = logging.getLogger()
        # Generate a unique identifier for the document's vectorstore file
        vectorstore_file = os.path.join(
            self.vectorstore_dir + "_" + self._text_file_type,
            f"{company_year}.vectorstore",
        )

        # Check if the vectorstore already exists
        if os.path.exists(vectorstore_file):
            logger.info(
                f"Vectorstore for {company_year}.{self._text_file_type} "
                f"already exists. Skipping."
            )
            return None

        # Measure time to create vectorstore
        start_time = time.time()

        # Load the TXT file and divide it into chunks
        try:
            chunks = self._divide_text_into_chunks(company_year)
        except Exception:
            logger.warning(
                f"Could not chunk text file for company year {company_year}."
                "Skipping."
            )
            return None
        logger.info(
            f"Number of chunks for {company_year}.{self._text_file_type}: "
            f"{len(chunks)}"
        )

        if self._embedding_model_host == "TogetherAI":
            # Set Together API key as environment variable
            os.environ["TOGETHER_API_KEY"] = self._together_api_key
            embedding_model = TogetherEmbeddings(
                model=self._embedding_model_name,
            )
        elif self._embedding_model_host == "HuggingFace":
            embedding_model = HuggingFaceEmbeddings(
                model_name=self._embedding_model_name
            )
        else:
            logger.warning(
                "Invalid embedding model host specified. Returning None."
            )
            return None

        # Create the vectorstore from the TXT chunks
        try:
            vectorstore = FAISS.from_documents(
                chunks,
                embedding_model,
            )
        except Exception as e:
            logger.warning(
                f"Exception occurred for document "
                f"{company_year}.{self._text_file_type}: {e}. Skipping."
            )
            return None

        # Save vectorstore to the dedicated directory
        vectorstore.save_local(vectorstore_file)
        logger.info(
            f"Document {company_year}.{self._text_file_type} stored in "
            f"vectorstore. Runtime: {round((time.time()-start_time), 2)} s"
        )
        del vectorstore
        del embedding_model
        torch.cuda.empty_cache()
        return 0

    def _divide_text_into_chunks(self, company_year: str) -> List[str]:
        """
        Divides the text document for a given company year into smaller chunks.

        This method loads the text document corresponding to the specified
        `company_year`, splits it into chunks using the
        `RecursiveCharacterTextSplitter` class, and returns the resulting
        chunks as a list of strings.

        Args:
            company_year (str): The company_year ID for the document being
            processed.

        Returns:
            List[str]: A list of text chunks obtained from the document.
        """
        # Specify the TXT/MD file path
        txt_file = os.path.join(
            self.txt_dir, f"{company_year}.{self._text_file_type}"
        )
        # Load the TXT/MD document
        if self._text_file_type == "md":
            raw_document = UnstructuredMarkdownLoader(
                txt_file, encoding="UTF-8"
            ).load()
            splitter = MarkdownTextSplitter(
                chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap
            )
            chunks = splitter.split_documents(raw_document)
            """
            separators = self._chunk_separators
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                separators=separators,
                add_start_index=True,  # Enables start index tracking
            )
            # Split the raw document into chunks
            chunks = text_splitter.split_documents(raw_document)
            """
        else:
            with open(txt_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_text = data["text"]
                metadata = data["metadata"]
                # Create a Document object from the concatenated text
                raw_document = [
                    Document(
                        page_content=all_text,
                        metadata={"source_info": metadata},
                    )
                ]
                chunks = self._split_into_chunks_with_metadata(
                    raw_document, metadata
                )

        return chunks

    def _rerank_context(
        self,
        reranking_model: CrossEncoder,
        query: str,
        context_list: List[str],
        top_k: int,
    ) -> List[str]:
        """
        Rerank a list of context passages based on their relevance to a query
        using a cross-encoder model.

        Args:
            reranking_model (CrossEncoder): The cross-encoder reranking model
            used to predict relevance scores.
            query (str): The query string for which the contexts are reranked.
            context_list (List[str]): A list of context passages to be
            reranked based on relevance to the query.
            top_k (int): The number of top relevant contexts to return after
            reranking.

        Returns:
            List[str]: A list of the top-k most relevant context passages,
            sorted by relevance.
        """
        # Prepare the inputs for the cross-encoder
        inputs = [(query, context) for context in context_list]
        # Get scores from the cross-encoder
        scores = reranking_model.predict(inputs)
        # Sort documents based on the scores
        reranked_context = [
            doc for _, doc in sorted(zip(scores, context_list), reverse=True)
        ]
        # Return the top-k documents
        if len(reranked_context) > top_k:
            return reranked_context[:top_k]
        else:
            return reranked_context

    def _split_into_chunks_with_metadata(self, raw_document, metadata):
        # Initialize RecursiveCharacterTextSplitter
        separators = self._chunk_separators
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=separators,
            add_start_index=True,  # Enables start index tracking
        )
        # Split the raw document into chunks
        chunks = text_splitter.split_documents(raw_document)

        for idx, chunk in enumerate(chunks):
            chunk_start = chunk.metadata["start_index"]
            chunk_end = chunk_start + len(chunk.page_content)
            chunk_metadata = []

            # Initialize a cumulative position tracker for each page
            cumulative_position = 0
            # Introduce variable to track chunk overlap
            chunk_page_overlap = 0

            for entry in metadata:
                # Calculate start and end positions for this page
                entry_start = cumulative_position
                entry_end = entry_start + entry["text_length"]

                # Update cumulative position for the next page
                cumulative_position = entry_end + 1

                # Check if this page intersects with the chunk's range
                if entry_start <= chunk_end and entry_end >= chunk_start:
                    chunk_page_overlap = 1
                    chunk_metadata.append(
                        {
                            "source_document": entry["source_document"],
                            "page_number": entry["page_number"],
                            "sequence_number": idx + 1,
                        }
                    )

            if chunk_page_overlap == 0:
                logging.warning(
                    "Chunk does not overlap with any pages. "
                    "No metadata generated."
                )
            # Update each chunk's metadata with the specific source pages
            chunk.metadata["source_info"] = chunk_metadata
        return chunks

    def _retrieve_parent_documents(
        self, retriever, context, document_name, data_point_description
    ):
        # Step 1: Extract and expand target sequence numbers with window size
        window_size = self._parent_document_window_size
        target_sequence_ranges = defaultdict(list)
        target_metadata = {}

        # Collect sequence numbers from context and expand by the window size
        for doc in context:
            try:
                target_sequence_number = doc.metadata["source_info"][0][
                    "sequence_number"
                ]
            except Exception as e:
                logging.warning(
                    f"Parent documents couldn't be fetched for {document_name}"
                    f" and {data_point_description}: {e}. Skipping document."
                )
                return None
            # Save the metadata of the target document for later use
            target_metadata[target_sequence_number] = doc.metadata
            # Store the expanded range for each target sequence number
            for offset in range(-window_size, window_size + 1):
                target_sequence_ranges[target_sequence_number].append(
                    target_sequence_number + offset
                )

        # Step 2: Retrieve matching documents by target sequence number
        matched_elements = defaultdict(list)

        if self._apply_hybrid_search:
            docstore_items = retriever.retrievers[
                0
            ].vectorstore.docstore._dict.items()
        else:
            docstore_items = retriever.vectorstore.docstore._dict.items()

        # Iterate over all documents in the docstore
        for _, doc in docstore_items:
            # Access 'source_info' in each document's metadata
            for source_info in doc.metadata.get("source_info", []):
                seq_num = source_info.get("sequence_number")
                # Check if this sequence number is within any target range
                for target_seq, seq_range in target_sequence_ranges.items():
                    if seq_num in seq_range:
                        matched_elements[target_seq].append(doc)
                        break  # Move to the next doc once a match is found

        # Step 3: Concatenate texts for each target sequence number
        concatenated_context = []

        for target_seq, docs in matched_elements.items():
            concatenated_text = " ".join([doc.page_content for doc in docs])
            # Retrieve the original target document's metadata
            original_metadata = target_metadata.get(target_seq, {})
            # Create a new Document for each concatenated text with metadata
            concatenated_context.append(
                Document(
                    page_content=concatenated_text, metadata=original_metadata
                )
            )

        return concatenated_context
