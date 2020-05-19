import time
import requests
import os
import warnings
import pkg_resources

from grid2op.MakeEnv.MakeFromPath import make_from_dataset_path
from grid2op.MakeEnv.MakeOld import make_old
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
    # keep the old names for now
    "case14_realistic": DEV_DATASET.format("rte_case14_realistic"),
    "case14_redisp": DEV_DATASET.format("rte_case14_redisp"),
    "case14_test": DEV_DATASET.format("rte_case14_test"),
    "case5_example": DEV_DATASET.format("rte_case5_example"),
    "case14_fromfile": DEV_DATASET.format("rte_case14_test")
}

_REQUEST_FAIL_EXHAUSTED_ERR = "Impossible to retrieve data at \"{}\".\n" \
                              "If the problem persists, please contact grid2op organizers"
_REQUEST_FAIL_RETRY_ERR = "Failure to get a reponse from the url \"{}\".\n" \
                          "Retrying.. {} attempt(s) remaining"
_REQUEST_EXCEPT_RETRY_ERR = "Exception in getting an answer from \"{}\".\n" \
                            "Retrying.. {} attempt(s) remaining"


def _send_request_retry(url, nb_retry=10, gh_session=None):
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
    except:
        warnings.warn(_REQUEST_EXCEPT_RETRY_ERR.format(url, nb_retry-1))
        time.sleep(1)
        return _send_request_retry(url, nb_retry=nb_retry-1, gh_session=gh_session)


# _LIST_REMOTE_URL = "https://api.github.com/repos/bdonnot/grid2op-datasets/contents/contents.json"
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
def _list_available_remote_env_aux():
    answer = _send_request_retry(_LIST_REMOTE_URL)
    try:
        answer_json = answer.json()
    except Exception as e:
        raise Grid2OpException(_LIST_REMOTE_INVALID_CONTENT_JSON_ERR.format(e))

    if not _LIST_REMOTE_KEY in answer_json:
        raise Grid2OpException(_LIST_REMOTE_CORRUPTED_CONTENT_JSON_ERR)
    time.sleep(1)
    avail_datasets = _send_request_retry(answer_json[_LIST_REMOTE_KEY])
    try:
        avail_datasets_json = avail_datasets.json()
    except Exception as e:
        raise Grid2OpException(_LIST_REMOTE_INVALID_DATASETS_JSON_ERR.format(e))
    return avail_datasets_json


_FETCH_ENV_UNKNOWN_ERR = "Impossible to find the environment named \"{}\".\n" \
                         "Current available environments are:\n{}"
# _FETCH_ENV_TAR_URL = "https://github.com/BDonnot/grid2op-datasets/releases/download/{}/{}.tar.bz2"


def _fecth_environments(dataset_name):
    avail_datasets_json = _list_available_remote_env_aux()
    if not dataset_name in avail_datasets_json:
        known_ds = sorted(avail_datasets_json.keys())
        raise UnknownEnv(_FETCH_ENV_UNKNOWN_ERR.format(dataset_name, known_ds))
    # url = _FETCH_ENV_TAR_URL.format(avail_datasets_json[dataset_name], dataset_name)
    dict_ =  avail_datasets_json[dataset_name]
    baseurl, filename = dict_["base_url"], dict_["filename"]
    url = baseurl + filename
    # name is "tar.bz2" so i need to get rid of 2 extensions
    ds_name_dl = os.path.splitext(os.path.splitext(filename)[0])[0]
    return url, ds_name_dl


_EXTRACT_DS_NAME_CONVERT_ERR = "The \"dataset_name\" argument " \
                               "should be convertible to string, " \
                               "but \"{}\" was provided."
_EXTRACT_DS_NAME_RECO_ERR = "Impossible to recognize the environment name from path \"{}\""


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


_MAKE_DEV_ENV_WARN = "You are using a development environment. " \
                     "This environment is not intended for training agents."
_MAKE_DEV_ENV_DEPRECATED_WARN = "Dev env \"{}\" has been deprecated " \
                                "and will be removed in future version.\n" \
                                "Please update to dev envs starting by \"rte\" or \"l2rpn\""
_MAKE_FIRST_TIME_WARN = "It is the first time you use the environment \"{}\".\n" \
                        "We will attempt to download this environment from remote"
_MAKE_UNKNOWN_ENV = "Impossible to load the environment named \"{}\"."


def make(dataset="rte_case14_realistic", test=False, **kwargs):
    """
    This function is a shortcut to rapidly create some (pre defined) environments within the grid2op Framework.

    Other environments, with different powergrids will be made available in the future and will be easily downloadable
    using this function.

    It mimic the `gym.make` function.

    Parameters
    ----------

    dataset: ``str``
        Path to the dataset folder, defaults to "rte_case14_realistic"

    test: ``bool``
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
    # dataset arg is a valid path: load it
    if os.path.exists(dataset):
        return make_from_dataset_path(dataset, **kwargs)

    # Not a path: get the dataset name and cache path
    dataset_name = _extract_ds_name(dataset)
    real_ds_path = os.path.join(grid2op.MakeEnv.PathUtils.DEFAULT_PATH_DATA, dataset_name)

    # Unknown dev env
    if test and dataset_name not in TEST_DEV_ENVS:
        raise Grid2OpException(_MAKE_UNKNOWN_ENV)

    # Known test env and test flag enabled
    if test:
        warnings.warn(_MAKE_DEV_ENV_WARN)
        # Warning for deprecated dev envs
        if not dataset_name.startswith("rte"):
            warnings.warn(_MAKE_DEV_ENV_DEPRECATED_WARN.format(dataset_name))
        return make_from_dataset_path(TEST_DEV_ENVS[dataset_name], **kwargs)

    # Env directory is present in the DEFAULT_PATH_DATA
    if os.path.exists(real_ds_path):
        return make_from_dataset_path(real_ds_path, **kwargs)

    # Env needs to be downloaded
    warnings.warn(_MAKE_FIRST_TIME_WARN.format(dataset_name))
    _create_path_folder(grid2op.MakeEnv.PathUtils.DEFAULT_PATH_DATA)
    url, ds_name_dl = _fecth_environments(dataset_name)
    _aux_download(url, dataset_name, grid2op.MakeEnv.PathUtils.DEFAULT_PATH_DATA, ds_name_dl)
    return make_from_dataset_path(real_ds_path, **kwargs)
