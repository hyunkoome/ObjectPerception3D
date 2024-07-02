# import logging
# import os
# def check_sequence_name_with_all_version(sequence_file):
#     if not os.path.exists(sequence_file):
#         found_sequence_file = sequence_file
#         for pre_text in ['training', 'validation', 'testing']:
#             if not os.path.exists(sequence_file):
#                 temp_sequence_file = str(sequence_file).replace('segment', pre_text + '_segment')
#                 if os.path.exists(temp_sequence_file):
#                     found_sequence_file = temp_sequence_file
#                     break
#         if not os.path.exists(found_sequence_file):
#             found_sequence_file = str(sequence_file).replace('_with_camera_labels', '')
#         if os.path.exists(found_sequence_file):
#             sequence_file = found_sequence_file
#     return sequence_file
#
#
# def create_logger(log_file=None, rank=0, log_level=logging.INFO):
#     logger = logging.getLogger(__name__)
#     logger.setLevel(log_level if rank == 0 else 'ERROR')
#     formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
#     console = logging.StreamHandler()
#     console.setLevel(log_level if rank == 0 else 'ERROR')
#     console.setFormatter(formatter)
#     logger.addHandler(console)
#     if log_file is not None:
#         file_handler = logging.FileHandler(filename=log_file)
#         file_handler.setLevel(log_level if rank == 0 else 'ERROR')
#         file_handler.setFormatter(formatter)
#         logger.addHandler(file_handler)
#     logger.propagate = False
#     return logger
#

import os
import logging
import typing


def search_in_versions(sequence_file: str) -> str:
    """
    Searches for a specific sequence file in different versions.
    :param sequence_file: The path of the sequence file.
    :return: The path of the sequence file in the found version, or the original sequence file path if not found.
    """

    for version in ['training', 'validation', 'testing']:
        temp_sequence_file = sequence_file.replace('segment', f'{version}_segment')
        if os.path.exists(temp_sequence_file):
            return temp_sequence_file
    return sequence_file


def check_sequence_name_with_all_version(sequence_file: str) -> str:
    """
    Check if the given sequence file exists and return the file path.
    If the file does not exist, search for the file in different versions and return the found file path.
    If the file still does not exist, remove the '_with_camera_labels' suffix from the sequence file name and check if the modified file exists.
    If the modified file exists, return its path. Otherwise, return the original sequence file path.

    :param sequence_file: A string representing the path to the sequence file
    :return: A string representing the path to the existing sequence file
    """
    if os.path.exists(sequence_file):
        return sequence_file
    found_sequence_file = search_in_versions(sequence_file)
    if not os.path.exists(found_sequence_file):
        found_sequence_file = sequence_file.replace('_with_camera_labels', '')
    return found_sequence_file if os.path.exists(found_sequence_file) else sequence_file


def setup_handler(handler: logging.Handler, level: typing.Union[str, int]) -> logging.Handler:
    """
    Setup Handler method

    :param handler: The logging handler to be setup
    :param level: The logging level to set for the handler
    :return: The setup logging handler

    """
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s'))
    return handler


def create_logger(rank: int = 0, log_level: int = logging.INFO,
                  log_file: typing.Optional[str] = None) -> logging.Logger:
    """
    Create a logger with specified parameters.

    :param rank: The rank of the logger. Default is 0.
    :param log_level: The log level of the logger. Default is logging.INFO.
    :param log_file: The path to the log file. Default is None.
    :return: The created logger.

    Example usage:
        logger = create_logger(0, logging.INFO, 'log_file.txt')
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    logger.addHandler(setup_handler(logging.StreamHandler(), log_level if rank == 0 else 'ERROR'))
    if log_file is not None:
        logger.addHandler(setup_handler(handler=logging.FileHandler(log_file, mode='w', encoding=None, delay=False),
                                        level=log_level if rank == 0 else 'ERROR'))
        logger.propagate = False
    return logger
