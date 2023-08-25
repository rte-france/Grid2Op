#!/usr/bin/env python

# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.


import argparse
import os
import sys

from grid2op.MakeEnv.PathUtils import DEFAULT_PATH_DATA
from grid2op.Download.DownloadDataset import main_download
from grid2op.Download.DownloadDataset import LI_VALID_ENV


def main(args=None):
    """
    DEPRECATED, use the `grid2op.make(...)`  function that will automatically download an environment
    if not present on your system already.
    """
    if args is None:
        args = download_cli()

    dataset_name = args.name
    try:
        path_data = os.path.abspath(args.path_save)
    except Exception as e:
        print(
            'Argument "--path_save" should be a valid path (directory) on your machine.'
        )
        sys.exit(1)

    try:
        main_download(dataset_name, path_data)
    except Exception as e:
        sys.exit("Aborted")


def download_cli():
    parser = argparse.ArgumentParser(
        description="Download some datasets compatible with grid2op."
    )
    parser.add_argument(
        "--path_save",
        default=DEFAULT_PATH_DATA,
        type=str,
        help="The path where the data will be downloaded.",
    )
    parser.add_argument(
        "--name",
        default="l2rpn_case14_sandbox",
        type=str,
        help="The name of the dataset (one of {} )." "".format(",".join(LI_VALID_ENV)),
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = download_cli()
    main(args)
