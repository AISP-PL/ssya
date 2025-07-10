import argparse
import logging

# Launch GUI application
from PyQt5.QtWidgets import QApplication
from yaya_tools import __version__

from ssya.gui import MainWindow

logger = logging.getLogger(__name__)


def logging_terminal_setup() -> None:
    """
    Setup logging for the application.

    Parameters
    ----------
    path_field : str
        Field in the config file that contains the path to the log file.
        Default is "path".
    is_terminal : bool
        If True, logs will be printed to the terminal.
        Default is True.
    """
    logging.getLogger().setLevel(logging.DEBUG)  # Ensure log level is set to DEBUG
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    logger.info("\n\n###### Logging start of terminal session ######\n")


def main() -> None:
    """
    Main function for dataset management

    Returns
    -------
    None
    """
    logging_terminal_setup()

    # Argument parser
    parser = argparse.ArgumentParser(add_help=False, description="YAYa dataset management tool")
    parser.add_argument("-i", "--dataset_path", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("-h", "--help", action="help", help="Show this help message and exit.")
    parser.add_argument("-v", action="version", version=__version__, help="Show version and exit.")
    args = parser.parse_args()

    dataset_path = args.dataset_path

    # Data loading done in GUI
    app = QApplication([])
    window = MainWindow(dataset_path)
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
