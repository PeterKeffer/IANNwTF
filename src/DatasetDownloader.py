import os
import urllib.request
import logging
from pathlib import Path

class DatasetDownloader:

    def __init__(self, url:str, output_filepath:str, override:bool=False):
        """

        :param url: source of dataset
        :param output_filepath: destination of downloaded files
        :param override: should existing dataset be overridden ?
        """
        if os.path.isfile(output_filepath) and not override:
            logging.warning("File already exists!")
            return True
        output_file_directory = os.path.dirname(output_filepath)
        Path(output_file_directory).mkdir(parents=True, exist_ok=True)
        self.download(url, output_filepath)

    def download(self, url:str, output_filename:str):
        logging.info("Download Started...")
        # TODO: Progressbar
        urllib.request.urlretrieve(url, output_filename)
        logging.info("Download Finished")

