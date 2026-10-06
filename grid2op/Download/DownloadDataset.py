# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import hashlib
import os
import sys
import warnings
from tqdm import tqdm
import re

import tarfile

from grid2op.Exceptions import Grid2OpException

try:
    import urllib.parse
    import urllib.request
except Exception as exc_:  # pragma: no cover
    raise RuntimeError(f"Impossible to find library urllib. Please install it, error was:\n{exc_}.")

URL_GRID2OP_DATA = "https://github.com/Tezirg/Grid2Op/releases/download/{}/{}"

DATASET_TAG_v0_1_0 = "datasets-v0.1.0"

DICT_URL_GRID2OP_DL = {
    "rte_case14_realistic": URL_GRID2OP_DATA.format(
        DATASET_TAG_v0_1_0, "rte_case14_realistic.tar.bz2"
    ),
    "rte_case14_redisp": URL_GRID2OP_DATA.format(
        DATASET_TAG_v0_1_0, "rte_case14_redisp.tar.bz2"
    ),
    "l2rpn_2019": URL_GRID2OP_DATA.format(DATASET_TAG_v0_1_0, "l2rpn_2019.tar.bz2"),
}

LI_VALID_ENV = sorted(['"{}"'.format(el) for el in DICT_URL_GRID2OP_DL.keys()])

ALLOWED_SCHEMES = {"http", "https"}

# Archive extraction safety limits (S5042)
_MAX_UNCOMPRESSED_SIZE = 30 * 1024 * 1024 * 1024  # 30 GB
_MAX_COMPRESSION_RATIO = 100  # reject archives that expand more than 100×


_SHA256_BLOCKSIZE = 1024 * 1024

_CHECKSUM_MISMATCH_ERR = (
    'The archive downloaded from "{url}" does not match the expected SHA-256 checksum '
    "(expected {expected}, got {got}). The file was deleted and nothing was extracted. "
    "The download may have been corrupted or the remote file may have been modified. "
    "Please try again and, if the problem persists, open an issue at "
    "https://github.com/Grid2op/grid2op/issues"
)
_CHECKSUM_MISSING_WARN = (
    'No SHA-256 checksum is available for the archive of "{dataset_name}", its integrity '
    "cannot be verified before extraction."
)


def _sha256_file(path, blocksize=_SHA256_BLOCKSIZE):
    """
    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    Returns the hexadecimal SHA-256 digest of the file located at ``path``.
    """
    hash_ = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(blocksize), b""):
            hash_.update(block)
    return hash_.hexdigest()


def _check_archive_sha256(path, expected_sha256, url=""):
    """
    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    Raises a :class:`grid2op.Exceptions.Grid2OpException` (and removes the file) if the SHA-256 of
    the archive at ``path`` is not ``expected_sha256`` (hexadecimal, case insensitive).
    """
    expected = str(expected_sha256).strip().lower()
    got = _sha256_file(path)
    if got != expected:
        os.remove(path)
        raise Grid2OpException(
            _CHECKSUM_MISMATCH_ERR.format(url=url, expected=expected, got=got)
        )


class DownloadProgressBar(tqdm):  # pragma: no cover
    """
    INTERNAL

     .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    This class is here to show the progress bar when downloading this dataset
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):  # pragma: no cover
    """
    INTERNAL

     .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    This function download the file located at 'url' and save it to 'output_path'

    Parameters
    ----------
    url: ``str``
        The url of the file to download

    output_path: ``str``
        The path where the data will be stored.
    """
    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError(
            f"Unsafe URL scheme '{scheme}'. Only {ALLOWED_SCHEMES} are allowed."
        )
        
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def _aux_download(
    url, dataset_name, path_data, ds_name_dl=None, sha256=None
):  # pragma: no cover
    """
    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    If ``sha256`` is provided, the downloaded archive is checked against it before anything is
    extracted. If it is ``None`` a warning is emitted and the archive is extracted unverified.
    """
    if ds_name_dl is None:
        ds_name_dl = dataset_name
    final_path = os.path.join(path_data, ds_name_dl)
    if os.path.exists(final_path):
        str_ = (
            'Downloading and extracting this data would create a folder "{final_path}" '
            "but this folder already exists. Either you already downloaded the data, in this case "
            "you can invoke the environment from a python script with:\n"
            '\t env = grid2op.make("{final_path}")\n'
            'Alternatively you can also delete the folder "{final_path}" from your computer and run this command '
            "again.\n"
            "Finally, you can download the data in a different folder by specifying (in a command prompt):\n"
            '\t grid2op.download --name "{env_name}" --path_save PATH\\WHERE\\YOU\\WANT\\TO\\DOWNLOAD'
            "".format(final_path=final_path, env_name=dataset_name)
        )
        print(str_)
        raise Grid2OpException(str_)

    if not os.path.exists(path_data):
        print(
            'Creating path "{}" where data for "{}" environment will be downloaded.'
            "".format(path_data, ds_name_dl)
        )
        try:
            os.mkdir(path_data)
        except Exception as exc_:
            str_ = (
                'Impossible to create path "{}" to store the data. Please save the data in a different repository '
                'with setting the argument "--path_save"'
                "Error was:\n{}".format(path_data, exc_)
            )
            raise Grid2OpException(str_)

    base_dir = os.path.realpath(path_data)
    output_path = os.path.realpath(
        os.path.join(base_dir, "{}.tar.bz2".format(ds_name_dl))
    )
    if os.path.dirname(output_path) != base_dir:
        raise Grid2OpException(
            'The name "{}" of the dataset would write the archive outside of "{}", '
            "this is not allowed.".format(ds_name_dl, path_data)
        )

    # download the data (with progress bar)
    print("downloading the training data, this may take a while.")
    download_url(url, output_path)

    if sha256 is None:
        warnings.warn(_CHECKSUM_MISSING_WARN.format(dataset_name=dataset_name))
    else:
        _check_archive_sha256(output_path, sha256, url=url)

    tar = tarfile.open(output_path, "r:bz2")
    compressed_size = os.path.getsize(output_path)
    total_uncompressed = sum(m.size for m in tar.getmembers())
    if total_uncompressed > _MAX_UNCOMPRESSED_SIZE:
        tar.close()
        raise Grid2OpException(
            f"Refusing to extract archive: uncompressed size ({total_uncompressed} bytes) "
            f"exceeds the limit of {_MAX_UNCOMPRESSED_SIZE} bytes."
        )
    if compressed_size > 0 and total_uncompressed / compressed_size > _MAX_COMPRESSION_RATIO:
        tar.close()
        raise Grid2OpException(
            f"Refusing to extract archive: compression ratio "
            f"({total_uncompressed / compressed_size:.1f}x) exceeds {_MAX_COMPRESSION_RATIO}x."
        )
    print('Extract the tar archive in "{}"'.format(os.path.abspath(path_data)))
    tar.extractall(path_data)
    tar.close()

    # rename the file if necessary
    if ds_name_dl != dataset_name:
        try:
            os.rename(final_path, os.path.join(path_data, dataset_name))
        except FileNotFoundError as exc_:  # noqa: F841
            # the try catch is added because for some environments, the
            # archive name does not match the env name.
            # so the folder cannot be deleted properly.
            print(f"WARN: file \"{final_path}\" could not be found. You might need to manually delete it.")
            pass

    # and rm the tar bz2
    # bug in the AWS file... named ".tar.tar.bz2" ...
    os.remove(output_path)

    # check for update (if any)
    from grid2op.MakeEnv.UpdateEnv import _update_files

    _update_files(dataset_name)

    print(
        'You may now use the environment "{}" with the available data by invoking:\n'
        '\tenv = grid2op.make("{}")'
        "".format(dataset_name, dataset_name)
    )


def main_download(dataset_name, path_data):  # pragma: no cover
    """
    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    """
    dataset_name = dataset_name.lower().rstrip().lstrip()
    dataset_name = re.sub('"', "", dataset_name)

    if dataset_name not in DICT_URL_GRID2OP_DL:
        print(
            'Impossible to find environment named "{env_name}". Known environments are:\n{li_env}'
            "".format(env_name=dataset_name, li_env=",".join(LI_VALID_ENV))
        )
        sys.exit(1)

    url = DICT_URL_GRID2OP_DL[dataset_name]
    _aux_download(url, dataset_name, path_data)
