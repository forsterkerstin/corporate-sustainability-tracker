import multiprocessing

from post_processor import PostProcessor
from retrieval_augmented_generator import RetrievalAugmentedGenerator
from text_extractor import TextExtractor
from utils.setup_utils import load_config, set_up_logging
from vectorstore_manager import VectorstoreManager


def main():
    """
    Main function to orchestrate the download of company reports, text
    extraction, vectorstore creation, data point extraction using
    Retrieval-Augmented Generation (RAG), and post-processing of results.
    """
    # Set multiprocessing spawn method
    multiprocessing.set_start_method("spawn")
    # Load configuration file
    config = load_config()
    # Set up logging functionality
    log_filename = set_up_logging(config)

    # Create TextExtractor to extract text from the reports and save as TXT
    text_extractor = TextExtractor(config, log_filename)
    text_extractor.extract_and_save_text_concurrently()

    # Create VectorstoreManager to create and save vectorstores
    vectorstore_manager = VectorstoreManager(config, log_filename)
    vectorstore_manager.create_vectorstores_concurrently()

    # Create RetrievalAugmentedGenerator to extract data points using RAG
    retrieval_augmented_generator = RetrievalAugmentedGenerator(
        config, log_filename, vectorstore_manager
    )
    results_subdir = (
        retrieval_augmented_generator.extract_datapoints_concurrently()
    )

    # Create PostProcessor to process and save the final results
    post_processor = PostProcessor(results_subdir)
    post_processor.process_and_save_results()


if __name__ == "__main__":
    main()
