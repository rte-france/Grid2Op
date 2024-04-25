# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import time
import os
import re


import grid2op.MakeEnv.PathUtils
from grid2op.Exceptions import UnknownEnv
from grid2op.MakeEnv.UserUtils import list_available_local_env
from grid2op.MakeEnv.Make import _retrieve_github_content

_LIST_REMOTE_URL = (
    "https://api.github.com/repos/bdonnot/grid2op-datasets/contents/updates.json"
)
_LIST_REMOTE_ENV_HASH = (
    "https://api.github.com/repos/bdonnot/grid2op-datasets/contents/env_hashes.json"
)


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
    INTERNAL

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
    print(
        '\t Successfully updated file "{}" for environment "{}"'.format(
            file_name, env_name
        )
    )


def _do_env_need_update(env_name, env_hashes):
    if env_name not in env_hashes:
        # no hash for this environment is provided, i don't know, so in doubt i need to update it (old behaviour)
        return True
    else:
        # i check if "my" hash is different from the remote hash
        base_path = grid2op.get_current_local_dir()
        hash_remote_hex = env_hashes[env_name]
        hash_local = _hash_env(os.path.join(base_path, env_name))
        hash_local_hex = hash_local.hexdigest()
        res = hash_remote_hex != hash_local_hex
        return res


def _update_files(env_name=None, answer_json=None, env_hashes=None):
    """
    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    Update all the "modified" files of a given environment. If ``None`` is provided as input, all local environments
    will be checked for update.

    Parameters
    ----------
    env_name: ``str``
        Name of the environment you want to update (should be locally available)

    """
    avail_envs = list_available_local_env()

    if answer_json is None:
        # optimization to retrieve only once this file
        answer_json = _retrieve_github_content(_LIST_REMOTE_URL)

    if env_hashes is None:
        # optimization to retrieve only once this file
        env_hashes = _retrieve_github_content(_LIST_REMOTE_ENV_HASH)

    if env_name is None:
        # i update all the files for all the environments
        for env_name in avail_envs:
            _update_files(env_name, answer_json=answer_json, env_hashes=env_hashes)
    else:
        # i update the files for only an environment
        if env_name in avail_envs:
            need_update = _do_env_need_update(env_name, env_hashes)
            if env_name in answer_json and need_update:
                dict_main = answer_json[env_name]
                for k, dict_ in dict_main.items():
                    _update_file(dict_, env_name, file_name=k)
            elif need_update and env_name not in answer_json:
                print(
                    f'Environment: "{env_name}" is not up to date, but we did not found any files to update. '
                    f'IF this environment is officially supported by grid2op (see full list at '
                    f'https://grid2op.readthedocs.io/en/latest/available_envs.html#description-of-some-environments) '
                    f'Please write an issue at :\n\t\t'
                    f'https://github.com/rte-france/Grid2Op/issues/new?assignees=&labels=question&title=Environment%20{env_name}%20is%20not%20up%20to%20date%20but%20I%20cannot%20update%20it.&body=%3c%21%2d%2dDescribe%20shortly%20the%20context%20%2d%2d%3e%0d'
                )
            else:
                # environment is up to date
                print('Environment "{}" is up to date'.format(env_name))
        else:
            raise UnknownEnv(
                'Impossible to locate the environment named "{}". Have you downlaoded it?'
                "".format(env_name)
            )


def _aux_get_hash_if_none(hash_=None):
    """Auxilliary function used to avoid copy pasting the `hash_ = hashlib.blake2b()` part and that can
    be further changed if another hash is better later.
    
    Do not modify unless you have a good reason too.
    """
    if hash_ is None:
        # we use this as it is supposedly faster than md5
        # we don't really care about the "secure" part of it (though it's a nice tool to have)
        import hashlib  # lazy import
        hash_ = hashlib.blake2b()
    return hash_
    

def _aux_update_hash_text(text_, hash_=None):
    hash_ = _aux_get_hash_if_none(hash_)
    text_ = re.sub("\s", "", text_)
    hash_.update(text_.encode("utf-8"))
    return hash_
    

def _aux_hash_file(full_path_file, hash_=None):
    hash_ = _aux_get_hash_if_none(hash_)
    with open(full_path_file, "r", encoding="utf-8") as f:
        text_ = f.read()
        # this is done to ensure a compatibility between platform
        # sometime git replaces the "\r\n" in windows with "\n" on linux / macos and it messes
        # up the hash
        _aux_update_hash_text(text_, hash_)
    return hash_


# TODO make that a method of the environment maybe ?
def _hash_env(
    path_local_env,
    hash_=None,
    blocksize=64,  # TODO is this correct ?
):
    hash_ = _aux_get_hash_if_none(hash_)
    if os.path.exists(os.path.join(path_local_env, ".multimix")):
        # this is a multi mix, so i need to run through all sub env
        mixes = sorted(os.listdir(path_local_env))
        for mix in mixes:
            mix_dir = os.path.join(path_local_env, mix)
            if os.path.isdir(mix_dir):
                hash_ = _hash_env(mix_dir, hash_=hash_, blocksize=blocksize)
    else:
        # i am hashing a regular environment
        # first i hash the config files
        for fn_ in [
            "alerts_info.json",
            "config.py",
            "difficulty_levels.json",
            "grid.json",
            "grid_layout.json",
            "prods_charac.csv",
            "storage_units_charac.csv",
            # chronix2grid files, if any
            "loads_charac.csv",
            "params.json",
            "params_load.json",
            "params_loss.json",
            "params_opf.json",
            "params_res.json",
            "scenario_params.json",
        ]:  # list the file we want to hash (we don't hash everything
            full_path_file = os.path.join(path_local_env, fn_)

            if os.path.exists(full_path_file):
                _aux_hash_file(full_path_file, hash_)

        # now I hash the chronics
        # but as i don't want to read every chronics (for time purposes) i will only hash the names
        # of all the chronics
        path_chronics = os.path.join(path_local_env, "chronics")
        for chron_name in sorted(os.listdir(path_chronics)):
            hash_.update(chron_name.encode("utf-8"))
    return hash_
