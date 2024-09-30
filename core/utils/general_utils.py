import logging
import os
from box import Box

# Load the config file into box object
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "util_configs.yaml")
configs = Box.from_yaml(filename=config_path)


def create_logger(
    module_name: str,
    logger_format: str = configs.logger_objects.logger_format,
    logger_date_format: str = configs.logger_objects.logger_date_format,
) -> logging.Logger:
    """
    Create a customized logger object for the current module run.

        Args:
            module_name: Name of the calling module.
            logger_format: Information and format of each logger entry.
            logger_date_format: Date format of the logger entries.

        Returns:
            logger: The logger object.
    """
    # TODO: Add file handler to logger object when run folder is implemented

    # Create a custom logger object
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)

    # Create and configure stream handler (printing to the console)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_format = logging.Formatter(logger_format, datefmt=logger_date_format)
    stream_handler.setFormatter(stream_format)

    # Add handler to the logger object
    logger.addHandler(stream_handler)

    return logger
