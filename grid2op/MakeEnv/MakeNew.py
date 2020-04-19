import time
import requests
import os
import warnings
import pkg_resources

from grid2op.MakeEnv.MakeEnv import make_from_dataset_path
from grid2op.Exceptions import Grid2OpException, UnknownEnv
import grid2op.MakeEnv.PathUtils
# do not do "from grid2op.MakeEnv.PathUtils import DEFAULT_PATH_DATA" because this path
# can be modified by "UserUtils" If it's modified by the user, after it has been imported here
# the change will not be made.
from grid2op.MakeEnv.PathUtils import _create_path_folder

from grid2op.Download.DownloadDataset import _aux_download
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


# TODO add some kind of md5sum (or something else) in the remote json to check if version is locally updated (once in a while)
# TODO: logger instead of print would be so much better.
# TODO and an overall logger verbosity
def make_new(dataset_path="rte_case14_realistic", local=False, __dev=False, **kwargs):
    """
    This function is a shortcut to rapidly create some (pre defined) environments within the grid2op Framework.

    Other environments, with different powergrids will be made available in the future and will be easily downloadable
    using this function.

    It mimic the `gym.make` function.

    Parameters
    ----------

    dataset_path: ``str``
        Path to the dataset folder, defaults to "rte_case14_realistic"

    local: ``bool``
        Do not attempt to download data for environments stored remotely. See the global help of this module for
        detailed behavior above this flag.

    param: ``grid2op.Parameters.Parameters``, optional
        Type of parameters used for the Environment. Parameters defines how the powergrid problem is cast into an
        markov decision process, and some internal

    backend: ``grid2op.Backend.Backend``, optional
        The backend to use for the computation. If provided, it must be an instance of :class:`grid2op.Backend.Backend`.

    action_class: ``type``, optional
        Type of BaseAction the BaseAgent will be able to perform.
        If provided, it must be a subclass of :class:`grid2op.BaseAction.BaseAction`

    observation_class: ``type``, optional
        Type of BaseObservation the BaseAgent will receive.
        If provided, It must be a subclass of :class:`grid2op.BaseAction.BaseObservation`

    reward_class: ``type``, optional
        Type of reward signal the BaseAgent will receive.
        If provided, It must be a subclass of :class:`grid2op.BaseReward.BaseReward`

    gamerules_class: ``type``, optional
        Type of "Rules" the BaseAgent need to comply with. Rules are here to model some operational constraints.
        If provided, It must be a subclass of :class:`grid2op.RulesChecker.BaseRules`

    data_feeding_kwargs: ``dict``, optional
        Dictionnary that is used to build the `data_feeding` (chronics) objects.

    chronics_class: ``type``, optional
        The type of chronics that represents the dynamics of the Environment created. Usually they come from different
        folders.

    data_feeding: ``type``, optional
        The type of chronics handler you want to use.

    volagecontroler_class: ``type``, optional
        The type of :class:`grid2op.VoltageControler.VoltageControler` to use, it defaults to

    chronics_path: ``str``
        Path where to look for the chronics dataset (optional)

    grid_path: ``str``, optional
        The path where the powergrid is located.
        If provided it must be a string, and point to a valid file present on the hard drive.

    Returns
    -------
    env: :class:`grid2op.Environment.Environment`
        The created environment.
    """
    if os.path.exists(dataset_path):
        # first, if a path is provided, check if there is something there
        res = make_from_dataset_path(dataset_path, **kwargs)
        return res
    dataset_name = _extract_ds_name(dataset_path)
    res = None
    if os.path.exists(os.path.join(grid2op.MakeEnv.PathUtils.DEFAULT_PATH_DATA, dataset_name)):
        # second check in the DEFAULT_PATH_DATA if the environment is present (typically if env is given by name)
        real_ds_path = os.path.join(grid2op.MakeEnv.PathUtils.DEFAULT_PATH_DATA, dataset_name)
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
        print("It is the first time you use the environment \"{}\". We will attempt to download this environment from "
              "remote")
        _create_path_folder(grid2op.MakeEnv.PathUtils.DEFAULT_PATH_DATA)
        url = _fecth_environments(dataset_name)
        _aux_download(url, dataset_name, grid2op.MakeEnv.PathUtils.DEFAULT_PATH_DATA)
        real_ds_path = os.path.join(grid2op.MakeEnv.PathUtils.DEFAULT_PATH_DATA, dataset_name)
        res = make_from_dataset_path(real_ds_path, **kwargs)
    else:
        raise UnknownEnv("Impossible to load locally the environment named \"{}\".".format(dataset_path))
    return res