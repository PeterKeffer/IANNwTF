import os
import urllib.request
import logging
import zipfile
from pathlib import Path

class DatasetDownloader:

    def __init__(self, url: str, output_filepath: str, override: bool = False):
        """

        :param url: source of dataset
        :param output_filepath: destination of downloaded files
        :param override: should existing dataset be overridden ?
        """
        self.filepath = output_filepath

        output_file_directory = os.path.dirname(output_filepath)
        Path(output_file_directory).mkdir(parents=True, exist_ok=True)
        if os.path.isfile(output_filepath) and not override:
            logging.warning("File already exists!")
            return
        self.download(url, output_filepath)

    def download(self, url: str, output_filepath: str):
        logging.info("Download Started...")
        # TODO: Progressbar
        urllib.request.urlretrieve(url, output_filepath)
        logging.info("Download Finished")

    def unpack(self, output_filepath: str):
        with zipfile.ZipFile(self.filepath, 'r') as zip_reference:
            zip_reference.extractall(output_filepath)


