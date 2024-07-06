import os
import typing
import logging


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
    """
        logging.Formatter('%(asctime)s %(levelname)5s %(message)s')
        ==> logging.Formatter 객체를 생성할 때 사용되며,
            이 객체는 logging 핸들러(예: 파일 핸들러, 콘솔 핸들러 등)에 설정하여
             로그 메시지가 해당 형식에 맞게 출력되도록 함

        %(asctime)s: 로깅이 발생한 시간을 나타내는 부분입니다. 기본적으로 ISO 8601 형식(YYYY-MM-DDTHH:MM.mmmZ)으로 표시
        %(levelname)5s: 로깅 레벨을 나타내는 부분.
                        여기서 5s는 최소 너비 5를 지정하여 로깅 레벨을 오른쪽 정렬하여 표시
                        예를 들어, INFO, DEBUG와 같은 로깅 레벨이 들어감
        %(message)s: 실제 로깅 메시지를 나타내는 부분
                     이 부분은 실제 로그에 기록된 메시지 내용을 출력

        따라서 이 포맷 문자열을 사용하면, 로깅 메시지가 다음과 같은 형식으로 출력될 수 있음
        2024-07-04 12:34:56,789  INFO  This is a logging message
        ==> 2024-07-04 12:34:56,789: 로깅이 발생한 시간
            INFO: 로깅 레벨
            This is a logging message: 실제 로깅 메시지
    """
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
    logger.addHandler(
        hdlr=setup_handler(handler=logging.StreamHandler(),
                           level=log_level if rank == 0 else 'ERROR')
    )

    if log_file is not None:
        logger.addHandler(
            hdlr=setup_handler(handler=logging.FileHandler(log_file, mode='w', encoding=None, delay=False),
                               level=log_level if rank == 0 else 'ERROR')
        )
        logger.propagate = False

    return logger
