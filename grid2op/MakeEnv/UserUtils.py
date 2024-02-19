# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import json

from grid2op.MakeEnv.Make import _list_available_remote_env_aux
import grid2op.MakeEnv.PathUtils
from grid2op.MakeEnv.PathUtils import DEFAULT_PATH_CONFIG, KEY_DATA_PATH
from grid2op.Exceptions import Grid2OpException


def list_available_remote_env():
    """
    This function returns the list of available environments. It returns all the environments, you might already
    have downloaded some, they will be listed here.

    Returns
    -------
    res: ``list``
        a sorted list of available to environments that can be downloaded.

    Examples
    ---------
    A usage example is

    .. code-block:: python

        import grid2op
        li = grid2op.list_available_remote_env()
        li_fmt = '\\n * '.join(li)
        print(f"The available environments are: \\n * {li_fmt}")

    """
    avail_datasets_json = _list_available_remote_env_aux()
    return sorted(avail_datasets_json.keys())


def list_available_local_env():
    """
    This function returns the environment that are available locally. It does not return the environments that
    are included in the package.

    Returns
    -------
    res: ``list``
        a sorted list of available environments locally.

    Examples
    ---------

    .. code-block:: python

        import grid2op
        li = grid2op.list_available_local_env()
        li_fmt = '\\n + '.join(li)
        print(f"The locally available environments (without downloading anything) are: \\n * {li_fmt}")

    """
    res = []
    if not os.path.exists(grid2op.MakeEnv.PathUtils.DEFAULT_PATH_DATA):
        return res

    for el in os.listdir(grid2op.MakeEnv.PathUtils.DEFAULT_PATH_DATA):
        tmp_dir = os.path.join(grid2op.MakeEnv.PathUtils.DEFAULT_PATH_DATA, el)
        if (
            os.path.exists(os.path.join(tmp_dir, "config.py"))
            and os.path.exists(os.path.join(tmp_dir, "grid.json"))
        ) or os.path.exists(os.path.join(tmp_dir, ".multimix")):
            res.append(el)
    return res


def list_available_test_env():
    """
    This functions list the environment available through "grid2op.make(..., test=True)", which are the environment
    used for testing purpose, but available without the need to download any data.

    The "test" environment are provided with the grid2op package.

    Returns
    -------
    res: ``list``
        a sorted list of available environments for testing / illustration purpose.

    Examples
    ---------

    .. code-block:: python

        import grid2op
        li = grid2op.list_available_test_env()

        env = grid2op.make(li[0], test=True)

    """
    from grid2op.MakeEnv.Make import TEST_DEV_ENVS
    import re

    res = sorted(
        [
            el
            for el in TEST_DEV_ENVS.keys()
            if re.match("(^rte_.*)|(^l2rpn_.*)|(^educ_.*)", el) is not None
        ]
    )
    return res


def get_current_local_dir():
    """
    This function allows you to get the directory in which grid2op will download the datasets. This path can
    be modified with the ".grid2opconfig.json" file.

    Returns
    -------
    res: ``str``
        The current path were data are downloaded in.

    Examples
    ---------

    .. code-block:: python

        import grid2op
        print(f"Data about grid2op downloaded environments are stored in: \"{grid2op.get_current_local_dir()}\"")

    """
    return os.path.abspath(grid2op.MakeEnv.PathUtils.DEFAULT_PATH_DATA)


def change_local_dir(new_path):
    """
    This function will change the path were datasets are read to / from.

    The previous datasets will be left in the previous configuration folder and will not be accessible by other
    grid2op function such as "make" for example.

    Parameters
    ----------
    new_path: ``str``
        The new path in which to download the datasets.

    Examples
    ---------
    To set the download path, and the path where grid2op will look for available local environment you can:

    .. code-block:: python

        import grid2op
        local_dir = ...  # should be a valid path on your machine
        grid2op.change_local_dir(local_dir)

        # check it has worked:
        print(f"Data about grid2op downloaded environments are now stored in: \"{grid2op.get_current_local_dir()}\"")

    """

    try:
        new_path = str(new_path)
    except Exception as exc_:
        raise Grid2OpException(
            'The new path should be convertible to str. It is currently "{}"'.format(
                new_path
            )
        ) from exc_

    root_dir = os.path.split(new_path)[0]
    if not os.path.exists(root_dir):
        raise Grid2OpException(
            'Data cannot be stored in "{}" as the base path of this directory ("{}") does '
            "not exists.".format(new_path, root_dir)
        )

    if not os.path.isdir(new_path):
        raise Grid2OpException(
            'Data cannot be stored in "{}" as it is a file and not a directory.'.format(
                new_path
            )
        )

    newconfig = {}

    if os.path.exists(DEFAULT_PATH_CONFIG):
        try:
            with open(DEFAULT_PATH_CONFIG, "r", encoding="utf-8") as f:
                newconfig = json.load(f)
        except Exception as exc_:
            raise Grid2OpException(
                'Impossible to read the grid2op configuration files "{}". Make sure it is a '
                'valid json encoded with "utf-8" encoding.'.format(DEFAULT_PATH_CONFIG)
            ) from exc_

    newconfig[KEY_DATA_PATH] = new_path

    try:
        with open(DEFAULT_PATH_CONFIG, "w", encoding="utf-8") as f:
            json.dump(fp=f, obj=newconfig, sort_keys=True, indent=4)
    except Exception as exc_:
        raise Grid2OpException(
            'Impossible to write the grid2op configuration files "{}". Make sure you have '
            "writing access to it.".format(DEFAULT_PATH_CONFIG)
        ) from exc_

    grid2op.MakeEnv.PathUtils.DEFAULT_PATH_DATA = new_path
