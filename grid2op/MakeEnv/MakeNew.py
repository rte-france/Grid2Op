import time
import requests
import os
import warnings
import pkg_resources

from grid2op.MakeEnv.MakeEnv import make_from_dataset_path
from grid2op.Exceptions import Grid2OpException, UnknownEnv
from grid2op.MakeEnv.PathUtils import DEFAULT_PATH_DATA, _create_path_folder

from grid2op.Download.DownloadDataset import main_download
import pdb

PAHT_DATA_FOLDER = pkg_resources.resource_filename("grid2op", "data")
PATH_DATASET = os.path.join(PAHT_DATA_FOLDER, "{}")
TEST_DEV_ENVS = {"blank": "",
                 "rte_case14_realistic": PATH_DATASET.format("rte_case14_realistic"),
                 "rte_case14_redisp": PATH_DATASET.format("rte_case14_redisp"),
                 "rte_case14_test": PATH_DATASET.format("rte_case14_test"),
                 "rte_case5_example": PATH_DATASET.format("rte_case5_example"),
                 # keep the old names for now
                 "case14_realistic": PATH_DATASET.format("rte_case14_realistic"),
                 "case14_redisp": PATH_DATASET.format("rte_case14_redisp"),
                 "case14_test": PATH_DATASET.format("rte_case14_test"),
                 "case5_example": PATH_DATASET.format("rte_case5_example"),
                 "case14_fromfile": PATH_DATASET.format("rte_case14_test")
                 }


def _send_request_retry(url, nb_retry=10, gh_session=None):
    if nb_retry == 0:
        raise Grid2OpException("Impossible to retrieve data at \"{}\". If the problem persists contact one of "
                               "grid2op developer".format(url))

    if gh_session is None:
        gh_session = requests.Session()

    try:
        response = gh_session.get(url=url)
        if response.status_code == 200:
            return response
        print("Failure to get a reponse from the url \"{}\". {} more attempt(s) will be performed"
              "".format(url, nb_retry-1))
        time.sleep(1)
        return _send_request_retry(url, nb_retry=nb_retry-1, gh_session=gh_session)
    except Grid2OpException:
        raise
    except:
        print("Exception in getting an answer from \"{}\". {} more attempt(s) will be performed"
              "".format(url, nb_retry-1))
        time.sleep(1)
        return _send_request_retry(url, nb_retry=nb_retry-1, gh_session=gh_session)


def _list_available_remove_env_aux():
    github_api = "https://api.github.com/"
    url = github_api + "repos/bdonnot/grid2op-datasets/contents/contents.json"
    answer = _send_request_retry(url)
    try:
        answer_json = answer.json()
    except Exception as e:
        raise Grid2OpException("Impossible to retrieve available datasets. File could not be converted to json. "
                               "The error was \n\"{}\"".format(e))
    if not "download_url" in answer_json:
        raise Grid2OpException("Corrupted json retrieved from github api. Please wait a few minutes and try again."
                               "If the error persist, contact one of grid2op organizer")
    time.sleep(1)
    avail_datasets = _send_request_retry(answer_json["download_url"])
    try:
        avail_datasets_json = avail_datasets.json()
    except Exception as e:
        raise Grid2OpException("Impossible to retrieve available datasets. File could not be converted to json. "
                               "The error was \n\"{}\"".format(e))
    return avail_datasets_json


def list_available_remove_env():
    avail_datasets_json = _list_available_remove_env_aux()
    return sorted(avail_datasets_json.keys())


def _fecth_environments(dataset_name):
    avail_datasets_json = _list_available_remove_env_aux()
    if not dataset_name in avail_datasets_json:
        raise UnknownEnv("Impossible to find the environment named \"{}\". Current available environments are:\n{}"
                         "".format(dataset_name, sorted(avail_datasets_json)))
    url = f'https://github.com/BDonnot/grid2op-datasets/releases/download' \
          f'/{avail_datasets_json[dataset_name]}/{dataset_name}.tar.bz2'
    return url


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
        raise Grid2OpException("The \"dataset_name\" argument pass to \"_extract_ds_name\" should be convertible to "
                               "string, but \"{}\" was provided.".format(dataset_path))

    try:
        dataset_name = os.path.split(dataset_path)[-1]
    except:
        raise UnknownEnv("Impossible to recognize the environment name from path \"{}\"".format(dataset_path))
    dataset_name = dataset_name.lower().rstrip().lstrip()
    dataset_name = os.path.splitext(dataset_name)[0]
    return dataset_name


# TODO add some kind of md5sum in the remote json to check if version is locally updated (once in a while)
def make_new(dataset_path="rte_case14_realistic", local=False, __dev=False, **kwargs):
    if os.path.exists(dataset_path):
        # first, if a path is provided, check if there is something there
        res = make_from_dataset_path(dataset_path, **kwargs)
        return res
    dataset_name = _extract_ds_name(dataset_path)
    res = None
    if os.path.exists(os.path.join(DEFAULT_PATH_DATA, dataset_name)):
        # second check in the DEFAULT_PATH_DATA if the environment is present (typically if env is given by name)
        real_ds_path = os.path.join(DEFAULT_PATH_DATA, dataset_name)
        res = make_from_dataset_path(real_ds_path, **kwargs)
    elif dataset_name in TEST_DEV_ENVS and (__dev or local):
        if not __dev:
            warnings.warn("You are using a development environment. This is really not recommended for training agents.")
        if not dataset_name.startswith("rte"):
            warnings.warn("The name \"{}\" has been deprecated and will be removed in future version. Please update "
                          "with an environment names starting by \"rte\" or \"l2rpn\""
                          "".format(dataset_name))
        res = make_from_dataset_path(TEST_DEV_ENVS[dataset_name], **kwargs)
    elif not local:
        # third try to download the environment
        _create_path_folder(DEFAULT_PATH_DATA)
        url = _fecth_environments(dataset_name)
        main_download(url, DEFAULT_PATH_DATA)
        real_ds_path = os.path.join(DEFAULT_PATH_DATA, dataset_name)
        res = make_from_dataset_path(real_ds_path, **kwargs)
    else:
        raise UnknownEnv("Impossible to load locally the environment named \"{}\".".format(dataset_path))
    return res