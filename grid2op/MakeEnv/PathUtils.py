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
USE_CLASS_IN_FILE = False  # set to True for new behaviour (will be set to True in grid2op 1.11)


KEY_DATA_PATH = "data_path"
KEY_CLASS_IN_FILE = "class_in_file"
KEY_CLASS_IN_FILE_ENV_VAR = f"grid2op_{KEY_CLASS_IN_FILE}"

def str_to_bool(string: str) -> bool:
    """convert a "string" to a boolean, with the convention:
    
    - "t", "y", "yes", "true", "True", "TRUE" etc. returns True
    - "false", "False", "FALSE" etc. returns False
    - "1" returns True
    - "0" returns False
    
    """
    string_ = string.lower()
    if string_ in ["t", "true", "y", "yes", "on", "1"]:
        return True
    if string_ in ["f", "false", "n", "no", "off", "0"]:
        return False
    raise ValueError(f"Uknown way to convert `{string}` to a boolean. Please either set it to \"1\" or \"0\"")
    
        
if os.path.exists(DEFAULT_PATH_CONFIG):
    with open(DEFAULT_PATH_CONFIG, "r") as f:
        dict_ = json.load(f)

    if KEY_DATA_PATH in dict_:
        DEFAULT_PATH_DATA = os.path.abspath(dict_[KEY_DATA_PATH])
        
    if KEY_CLASS_IN_FILE in dict_:
        USE_CLASS_IN_FILE = bool(dict_[KEY_CLASS_IN_FILE])
        if KEY_CLASS_IN_FILE_ENV_VAR in os.environ:
            try:
                USE_CLASS_IN_FILE = str_to_bool(os.environ[KEY_CLASS_IN_FILE_ENV_VAR])
            except ValueError as exc:
                raise RuntimeError(f"Impossible to read the behaviour from `{KEY_CLASS_IN_FILE_ENV_VAR}` environment variable") from exc


def _create_path_folder(data_path):
    if not os.path.exists(data_path):
        try:
            os.mkdir(data_path)
        except Exception as exc_:
            raise RuntimeError(
                'Impossible to create a directory in "{}". Make sure you can write here. If you don\'t '
                'have writing permissions there, you can edit / create a config file in "{}"'
                'and set the "data_path" to point to a path where you can store data.'
                "".format(data_path, DEFAULT_PATH_CONFIG)
            )


def _aux_fix_backend_internal_classes(backend_cls, this_local_dir):
    # fix `my_bk_act_class` and `_complete_action_class`
    backend_cls._add_internal_classes(this_local_dir)
    tmp = {}
    backend_cls._make_cls_dict_extended(backend_cls, tmp, as_list=False)
