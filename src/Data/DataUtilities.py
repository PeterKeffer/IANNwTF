# Inspiration Source: https://stackoverflow.com/questions/56950987/download-file-from-url-and-save-it-in-a-folder-python
import os
from pathlib import Path

import requests


def download(origin: str, file_path: str) -> str:
    """
    Downloads an image and returns it's filepath
    :param origin: URL or Filepath(if Filepath change parameter file_path to None)
    :param file_path: Path where image should be saved
    :return: Filepath of saved Image
    """

    if (file_path == None or file_path == "") and os.path.exists(origin):
        return origin

    file_directory = os.path.dirname(file_path)
    Path(file_directory).mkdir(parents=True, exist_ok=True)

    r = requests.get(origin, stream=True)
    if r.ok:
        print("Saving to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
        return file_path
    else:  # HTTP status code 4XX/5XX
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))
