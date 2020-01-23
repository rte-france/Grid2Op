import os
import io
import sys
from tqdm import tqdm
import tarfile

import pdb
try:
    import urllib.request
except Exception as e:
    raise RuntimeError("Impossible to find library urllib. Please install it.")


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


if __name__ == "__main__":
    # todo add argparse

    url = "https://github.com/BDonnot/Grid2Op/releases/download/data_l2rpn_2019/data_l2rpn_2019.tar.bz2"
    path_data = "data"
    output_path = os.path.abspath(os.path.join(path_data, "data_l2rpn_2019.tar.bz2"))

    # download_url(url, output_path)
    # tarfile.TarFile.fileobject = get_file_progress_file_object_class(on_progress)

    # with tarfile.open(output_path, "r:bz2") as tar:
    # with tarfile.open(fileobj=ProgressFileObject(output_path)) as tar:

    # for el in tar:
    #    tar.extract(el, os.path.join(path_data, "{}".format(el)))
    #    pdb.set_trace()
    #    pbar.update(1)
    tar = tarfile.open(output_path, "r:bz2")
    print("Extract the tar archive in {}".format(path_data))
    tar.extractall(path_data)
    tar.close()
