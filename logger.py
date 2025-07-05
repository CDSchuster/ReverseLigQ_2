import logging


def setup_logging(logger_name, log_filename):

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler: only INFO (and optionally DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # Capture INFO and DEBUG
    console_handler.addFilter(lambda r: r.levelno < logging.WARNING)  # EXCLUDE WARNING and ERROR
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # File handler: only ERROR and above
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setLevel(logging.WARNING)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger