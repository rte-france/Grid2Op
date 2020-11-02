# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import time
import requests
import os
import warnings
import pkg_resources

from grid2op.MakeEnv.MakeFromPath import make_from_dataset_path, ERR_MSG_KWARGS
from grid2op.Exceptions import Grid2OpException, UnknownEnv
import grid2op.MakeEnv.PathUtils
from grid2op.MakeEnv.PathUtils import _create_path_folder
from grid2op.Download.DownloadDataset import _aux_download


DEV_DATA_FOLDER = pkg_resources.resource_filename("grid2op", "data")
DEV_DATASET = os.path.join(DEV_DATA_FOLDER, "{}")
TEST_DEV_ENVS = {
    "blank": DEV_DATASET.format("blank"),
    "rte_case14_realistic": DEV_DATASET.format("rte_case14_realistic"),
    "rte_case14_redisp": DEV_DATASET.format("rte_case14_redisp"),
    "rte_case14_test": DEV_DATASET.format("rte_case14_test"),
    "rte_case5_example": DEV_DATASET.format("rte_case5_example"),
    "rte_case118_example": DEV_DATASET.format("rte_case118_example"),
    "l2rpn_wcci_2020": DEV_DATASET.format("l2rpn_wcci_2020"),
    "rte_case14_opponent": DEV_DATASET.format("rte_case14_opponent"),
    "l2rpn_neurips_2020_track2": DEV_DATASET.format("l2rpn_neurips_2020_track2"),
    "l2rpn_neurips_2020_track1": DEV_DATASET.format("l2rpn_neurips_2020_track1"),
    # keep the old names for now
    "case14_realistic": DEV_DATASET.format("rte_case14_realistic"),
    "case14_redisp": DEV_DATASET.format("rte_case14_redisp"),
    "case14_test": DEV_DATASET.format("rte_case14_test"),
    "case5_example": DEV_DATASET.format("rte_case5_example"),
    "case14_fromfile": DEV_DATASET.format("rte_case14_test"),
    "educ_case14_redisp": DEV_DATASET.format("educ_case14_redisp"),
}

_REQUEST_FAIL_EXHAUSTED_ERR = "Impossible to retrieve data at \"{}\".\n" \
                              "If the problem persists, please contact grid2op developers by sending an issue at " \
                              "https://github.com/rte-france/Grid2Op/issues"
_REQUEST_FAIL_RETRY_ERR = "Failure to get a reponse from the url \"{}\".\n" \
                          "Retrying.. {} attempt(s) remaining"
_REQUEST_EXCEPT_RETRY_ERR = "Exception in getting an answer from \"{}\".\n" \
                            "Retrying.. {} attempt(s) remaining"

_LIST_REMOTE_URL = "https://api.github.com/repos/bdonnot/grid2op-datasets/contents/datasets.json"
_LIST_REMOTE_KEY = "download_url"
_LIST_REMOTE_INVALID_CONTENT_JSON_ERR = "Impossible to retrieve available datasets. " \
                                        "File could not be converted to json. " \
                                        "Parsing error:\n {}"
_LIST_REMOTE_CORRUPTED_CONTENT_JSON_ERR = "Corrupted json retrieved from github api. " \
                                         "Please wait a few minutes and try again. " \
                                         "If the error persist, contact grid2op organizers"
_LIST_REMOTE_INVALID_DATASETS_JSON_ERR = "Impossible to retrieve available datasets. " \
                                         "File could not be converted to json. " \
                                         "The error was \n\"{}\""

_FETCH_ENV_UNKNOWN_ERR = "Impossible to find the environment named \"{}\".\n" \
                         "Current available environments are:\n{}"

_MULTIMIX_FILE = ".multimix"

_MAKE_DEV_ENV_WARN = "You are using a development environment. " \
                     "This environment is not intended for training agents. It might not be up to date "\
                     "and its primary use if for tests (hence the \"test=True\" you passed as argument). "\
                     "Use at your own risk."
_MAKE_DEV_ENV_DEPRECATED_WARN = "Dev env \"{}\" has been deprecated " \
                                "and will be removed in future version.\n" \
                                "Please update to dev envs starting by \"rte\" or \"l2rpn\""
_MAKE_FIRST_TIME_WARN = "It is the first time you use the environment \"{}\".\n" \
                        "We will attempt to download this environment from remote"
_MAKE_UNKNOWN_ENV = "Impossible to load the environment named \"{}\"."

_EXTRACT_DS_NAME_CONVERT_ERR = "The \"dataset_name\" argument " \
                               "should be convertible to string, " \
                               "but \"{}\" was provided."
_EXTRACT_DS_NAME_RECO_ERR = "Impossible to recognize the environment name from path \"{}\""


def _send_request_retry(url, nb_retry=10, gh_session=None):
    """
    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
    """
    if nb_retry <= 0:
        raise Grid2OpException(_REQUEST_FAIL_EXHAUSTED_ERR.format(url))

    if gh_session is None:
        gh_session = requests.Session()

    try:
        response = gh_session.get(url=url)
        if response.status_code == 200:
            return response
        warnings.warn(_REQUEST_FAIL_RETRY_ERR.format(url, nb_retry-1))
        time.sleep(1)
        return _send_request_retry(url, nb_retry=nb_retry-1, gh_session=gh_session)
    except Grid2OpException:
        raise
    except KeyboardInterrupt:
        raise
    except:
        warnings.warn(_REQUEST_EXCEPT_RETRY_ERR.format(url, nb_retry-1))
        time.sleep(1)
        return _send_request_retry(url, nb_retry=nb_retry-1, gh_session=gh_session)


def _retrieve_github_content(url, is_json=True):
    answer = _send_request_retry(url)
    try:
        answer_json = answer.json()
    except Exception as e:
        raise Grid2OpException(_LIST_REMOTE_INVALID_CONTENT_JSON_ERR.format(e))

    if not _LIST_REMOTE_KEY in answer_json:
        raise Grid2OpException(_LIST_REMOTE_CORRUPTED_CONTENT_JSON_ERR)
    time.sleep(1)
    avail_datasets = _send_request_retry(answer_json[_LIST_REMOTE_KEY])
    if is_json:
        try:
            res = avail_datasets.json()
        except Exception as e:
            raise Grid2OpException(_LIST_REMOTE_INVALID_DATASETS_JSON_ERR.format(e))
    else:
        res = avail_datasets.text
    return res


def _list_available_remote_env_aux():
    return _retrieve_github_content(url=_LIST_REMOTE_URL)


def _fecth_environments(dataset_name):
    avail_datasets_json = _list_available_remote_env_aux()
    if not dataset_name in avail_datasets_json:
        known_ds = sorted(avail_datasets_json.keys())
        raise UnknownEnv(_FETCH_ENV_UNKNOWN_ERR.format(dataset_name, known_ds))
    # url = _FETCH_ENV_TAR_URL.format(avail_datasets_json[dataset_name], dataset_name)
    dict_ = avail_datasets_json[dataset_name]
    baseurl, filename = dict_["base_url"], dict_["filename"]
    url = baseurl + filename
    # name is "tar.bz2" so i need to get rid of 2 extensions
    ds_name_dl = os.path.splitext(os.path.splitext(filename)[0])[0]
    return url, ds_name_dl


def _extract_ds_name(dataset_path):
    """
    If a path is provided, clean it to have a proper datasetname.

    If a dataset name is already provided, then i just returns it.

    Parameters
    ----------
    dataset_path: ``str``
        The path in the form of a

    Returns
    -------
    dataset_name: ``str``
        The name of the dataset (all lowercase, without "." etc.)

    """

    try:
        dataset_path = str(dataset_path)
    except:
        raise Grid2OpException(_EXTRACT_DS_NAME_CONVERT_ERR.format(dataset_path))

    try:
        dataset_name = os.path.split(dataset_path)[-1]
    except:
        raise UnknownEnv(_EXTRACT_DS_NAME_RECO_ERR.format(dataset_path))
    dataset_name = dataset_name.lower().rstrip().lstrip()
    dataset_name = os.path.splitext(dataset_name)[0]
    return dataset_name


def _aux_is_multimix(dataset_path):
    if os.path.exists(os.path.join(dataset_path, _MULTIMIX_FILE)):
        return True
    return False


def _aux_make_multimix(dataset_path, **kwargs):
    # Local import to prevent imports loop
    from grid2op.Environment import MultiMixEnvironment

    return MultiMixEnvironment(dataset_path, **kwargs)


def make(dataset="rte_case14_realistic", test=False, _add_to_name="", **kwargs):
    """
    This function is a shortcut to rapidly create some (pre defined) environments within the grid2op Framework.

    Other environments, with different powergrids will be made available in the future and will be easily downloadable
    using this function.

    It mimic the `gym.make` function.

    Parameters
    ----------

    dataset: ``str``
        Name of the environment you want to create

    test: ``bool``
        Whether you want to use a test environment (**NOT** recommended). Use at your own risk.

    kwargs:
        Other keyword argument to give more control on the environment you are creating. See
        the Parameters information of the :func:`make_from_dataset_path`.

    _add_to_name:
        Internal, do not use

    Returns
    -------
    env: :class:`grid2op.Environment.Environment`
        The created environment.

    Examples
    --------

    If you want to create the environment "rte_case14_realistic":

    .. code-block: python

        import grid2op
        env = grid2op.make("rte_case14_realistic")
        # env implements the openai gym interface (env.step, env.render, env.reset etc.)

    **NB** the first time you type this command, the dataset (approximately 300 MB for this one) will be
    downloaded from the internet, sizes vary per dataset.

    """
    accepted_kwargs = ERR_MSG_KWARGS.keys() | {"dataset", "test"}
    for el in kwargs:
        if not el in accepted_kwargs:
            raise Grid2OpException("The keyword argument \"{}\" you provided is invalid. Possible keyword "
                                   "arguments to create environments are \"{}\"."
                                   "".format(el, sorted(accepted_kwargs)))
    # Select how to create the environment:
    # Default with make from path
    make_from_path_fn = make_from_dataset_path

    # dataset arg is a valid path: load it
    if os.path.exists(dataset):
        # Check if multimix from path
        if _aux_is_multimix(dataset):
            make_from_path_fn = _aux_make_multimix
        return make_from_path_fn(dataset_path=dataset, **kwargs)

    # Not a path: get the dataset name and cache path
    dataset_name = _extract_ds_name(dataset)
    real_ds_path = os.path.join(grid2op.MakeEnv.PathUtils.DEFAULT_PATH_DATA, dataset_name)

    # Unknown dev env
    if test and dataset_name not in TEST_DEV_ENVS:
        raise Grid2OpException(_MAKE_UNKNOWN_ENV.format(dataset))

    # Known test env and test flag enabled
    if test:
        warnings.warn(_MAKE_DEV_ENV_WARN)
        # Warning for deprecated dev envs
        if not (dataset_name.startswith("rte") or dataset_name.startswith("l2rpn") or dataset_name.startswith("educ")):
            warnings.warn(_MAKE_DEV_ENV_DEPRECATED_WARN.format(dataset_name))
        ds_path = TEST_DEV_ENVS[dataset_name]
        # Check if multimix from path
        if _aux_is_multimix(ds_path):
            make_from_path_fn = _aux_make_multimix

        return make_from_path_fn(dataset_path=ds_path, _add_to_name=_add_to_name, **kwargs)

    # Env directory is present in the DEFAULT_PATH_DATA
    if os.path.exists(real_ds_path):
        if _aux_is_multimix(real_ds_path):
            make_from_path_fn = _aux_make_multimix
        return make_from_path_fn(real_ds_path, **kwargs)

    # Env needs to be downloaded
    warnings.warn(_MAKE_FIRST_TIME_WARN.format(dataset_name))
    _create_path_folder(grid2op.MakeEnv.PathUtils.DEFAULT_PATH_DATA)
    url, ds_name_dl = _fecth_environments(dataset_name)
    _aux_download(url, dataset_name, grid2op.MakeEnv.PathUtils.DEFAULT_PATH_DATA, ds_name_dl)

    # Check if multimix from path
    if _aux_is_multimix(real_ds_path):
        make_from_path_fn = _aux_make_multimix
    return make_from_path_fn(dataset_path=real_ds_path, **kwargs)
