"""
This utility file helps downloading the data for some environments.

Data are stored as a github "release".

This script works on MacOs, Linux and windows.
"""
import os
import argparse
import io
import sys
from tqdm import tqdm
import re

import tarfile

import pdb

try:
    import urllib.request
except Exception as e:
    raise RuntimeError("Impossible to find library urllib. Please install it.")

URL = None
DEFAULT_PATH_DATA = os.path.expanduser("~/data_grid2op")
DICT_URL_GRID2OP_DL = {"l2rpn_2019":
                    ("https://github.com/BDonnot/Grid2Op/releases/download/data_l2rpn_2019/data_l2rpn_2019.tar.bz2",
                     "data_l2rpn_2019"),
            "case14_redisp": (
                "https://github.com/BDonnot/Grid2Op/releases/download/case14_redisp/case14_redisp.tar.bz2",
                "case14_redisp"
            ),
            "case14_realistic": (
                "https://github.com/BDonnot/Grid2Op/releases/download/case14_realistic/case14_realistic.tar.bz2",
                "case14_realistic"
            )
            }

LI_VALID_ENV = sorted(["\"{}\"".format(el) for el in DICT_URL_GRID2OP_DL.keys()])


class DownloadProgressBar(tqdm):
    """
    This class is here to show the progress bar when downloading this dataset
    """
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """
    This function download the file located at 'url' and save it to 'output_path'
    Parameters
    ----------
    url: ``str``
        The url of the file to download

    output_path: ``str``
        The path where the data will be stored.
    """
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def main_download(dataset_name, path_data):

    dataset_name = dataset_name.lower().rstrip().lstrip()
    dataset_name = re.sub('"', "", dataset_name)

    if dataset_name not in DICT_URL_GRID2OP_DL:
        print("Impossible to find environment named \"{env_name}\". Known environments are:\n{li_env}"
              "".format(env_name=dataset_name, li_env=",".join(LI_VALID_ENV)))
        sys.exit(1)

    URL, ds_name_dl = DICT_URL_GRID2OP_DL[dataset_name]
    final_path = os.path.join(path_data, ds_name_dl)
    if os.path.exists(final_path):
        print("Downloading and extracting this data would create a folder \"{final_path}\" "
              "but this folder already exists. Either you already downloaded the data, in this case "
              "you can invoke the environment from a python script with:\n"
              "\t env = grid2op.make(name_env=\"{env_name}\", chronics_path=\"{final_path}\")\n"
              "Alternatively you can also delete the folder \"{final_path}\" from your computer and run this command "
              "again.\n"
              "Finally, you can download the data in a different folder by specifying (in a command prompt):\n"
              "\t python -m grid2op.download --name \"{env_name}\" --path_save PATH\WHERE\YOU\WANT\TO\DOWNLOAD"
              "".format(final_path=final_path, env_name=dataset_name))
        sys.exit(1)

    if not os.path.exists(path_data):
        print("Creating path \"{}\" where data for \"{}\" environment will be downloaded."
              "".format(path_data, dataset_name))
        try:
            os.mkdir(path_data)
        except Exception as e:
            print("Impossible to create path \"{}\" to store the data. Please save the data in a different repository "
                  "with setting the argument \"--path_save\""
                  "".format(path_data))
            sys.exit(1)

    output_path = os.path.abspath(os.path.join(path_data, "{}.tar.bz2".format(ds_name_dl)))

    # download the data (with progress bar)
    print("downloading the training data, this may take a while.")
    download_url(URL, output_path)

    tar = tarfile.open(output_path, "r:bz2")
    print("Extract the tar archive in \"{}\"".format(os.path.abspath(path_data)))
    tar.extractall(path_data)
    tar.close()
    print("You may now use the environment \"{}\" with the available data by invoking:\n"
          "\tenv = grid2op.make(name_env=\"{}\", chronics_path=\"{}\")"
          "".format(dataset_name, dataset_name, final_path))
