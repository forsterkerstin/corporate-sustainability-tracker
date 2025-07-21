import logging
import os
from datetime import datetime

import yaml


def load_config(config_file="./configs/config.yaml"):
    """
    Load a configuration file in YAML format.
    """
    with open(config_file, "r") as file:
        return yaml.safe_load(file)


def set_up_logging(config):
    """
    Set up logging functionality using the "logging" library.
    """
    log_dir = config["log_dir"]
    # Create a directory for today's date
    today = datetime.now().strftime("%Y-%m-%d")
    log_subdir = os.path.join(log_dir, today)
    os.makedirs(log_subdir, exist_ok=True)

    # Create a unique log file name based on the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_subdir, f"log_{timestamp}.log")

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_filename, mode="a")  # Append to log file
        ],
    )

    # Log configuration of the current run
    logging.info(
        f"Starting run with the following configuration:\n{'\n'.join(
            f'{key}: {value}' for key, value in config.items()
            )}"
    )

    return log_filename
