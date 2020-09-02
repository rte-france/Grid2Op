# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import time

import os

import grid2op.MakeEnv.PathUtils
from grid2op.Exceptions import UnknownEnv
from grid2op.MakeEnv.UserUtils import list_available_local_env
from grid2op.MakeEnv.Make import _retrieve_github_content

_LIST_REMOTE_URL = "https://api.github.com/repos/bdonnot/grid2op-datasets/contents/updates.json"


def _write_file(path_local_env, new_config, file_name):
    with open(os.path.join(path_local_env, file_name), "w", encoding="utf-8") as f:
        f.write(new_config)


def update_env(env_name=None):
    """
    This function allows you to retrieve the latest version of the some of files used to create the
    environment.

    File can be for example "config.py" or "prod_charac.csv" or "difficulty_levels.json".

    Parameters
    ----------
    env_name: ``str``
        The name of the environment you want to update the config file (must be an environment you
        have already downloaded). If ``None`` it will look for updates for all the environments
        locally available.

    Examples
    --------
    Here is an example on how to for the update of your environments:

    .. code-block:: python

        import grid2op
        grid2op.update_env()
        # it will download the files "config.py" or "prod_charac.csv" or "difficulty_levels.json"
        # of your local environment to match the latest version available.

    """
    _update_files(env_name=env_name)


def _update_file(dict_, env_name, file_name):
    """
    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    Update a single file of a single environment.

    File can be for example "config.py" or "prod_charac.csv" or "difficulty_levels.json".
    """
    baseurl, filename = dict_["base_url"], dict_["filename"]
    url_ = baseurl + filename
    time.sleep(1)
    new_config = _retrieve_github_content(url_, is_json=False)
    path_local_env = os.path.join(grid2op.MakeEnv.PathUtils.DEFAULT_PATH_DATA, env_name)
    if os.path.exists(os.path.join(path_local_env, ".multimix")):
        # this is a multimix env ...
        mixes = os.listdir(path_local_env)
        for mix in mixes:
            mix_dir = os.path.join(path_local_env, mix)
            if os.path.exists(os.path.join(mix_dir, file_name)):
                # this is indeed a mix
                _write_file(mix_dir, new_config, file_name=file_name)
    else:
        _write_file(path_local_env, new_config, file_name=file_name)
    print("Successfully updated file \"{}\" for environment \"{}\"".format(file_name, env_name))


def _update_files(env_name=None):
    """
    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    Update all the "modified" files of a given environment. If ``None`` is provided as input, all local environments
    will be checked for update.

    Parameters
    ----------
    env_name: ``str``
        Name of the environment you want to update (should be locally available)

    """
    avail_envs = list_available_local_env()
    if env_name is None:
        for env_name in avail_envs:
            _update_files(env_name)
    else:
        if env_name in avail_envs:
            answer_json = _retrieve_github_content(_LIST_REMOTE_URL)

            if env_name in answer_json:
                dict_main = answer_json[env_name]
                for k, dict_ in dict_main.items():
                    _update_file(dict_, env_name, file_name=k)
            else:
                # environment is up to date
                print("Environment \"{}\" is up to date".format(env_name))
        else:
            raise UnknownEnv("Impossible to locate the environment named \"{}\". Have you downlaoded it?"
                             "".format(env_name))
