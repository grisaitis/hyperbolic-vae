import logging

from termcolor import colored

# https://chat.openai.com/c/8fc5a121-29f8-4141-9ade-88957d98587e


class ColoredFormatter(logging.Formatter):
    COLORS = {"WARNING": "yellow", "ERROR": "red", "CRITICAL": "red", "DEBUG": "green", "INFO": "white"}

    def format(self, record):
        log_message = super().format(record)
        return colored(log_message, self.COLORS.get(record.levelname))


if __name__ == "__main__":
    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create a handler and set its level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create a formatter and add it to the handler
    formatter = ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(ch)

    # Test logging
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
