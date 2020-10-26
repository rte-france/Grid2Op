# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

# this files present utilitary class for handling the folder where data are stored mainly
import os
import json

DEFAULT_PATH_CONFIG = os.path.expanduser("~/.grid2opconfig.json")
DEFAULT_PATH_DATA = os.path.expanduser("~/data_grid2op")
KEY_DATA_PATH = "data_path"

if os.path.exists(DEFAULT_PATH_CONFIG):
    with open(DEFAULT_PATH_CONFIG, "r") as f:
        dict_ = json.load(f)

    if KEY_DATA_PATH in dict_:
        DEFAULT_PATH_DATA = os.path.abspath(dict_[KEY_DATA_PATH])


def _create_path_folder(data_path):
    if not os.path.exists(data_path):
        try:
            os.mkdir(data_path)
        except:
            raise RuntimeError("Impossible to create a directory in \"{}\". Make sure you can write here. If you don't "
                               "have writing permissions there, you can edit / create a config file in \"{}\""
                               "and set the \"data_path\" to point to a path where you can store data."
                               "".format(data_path, DEFAULT_PATH_CONFIG))

