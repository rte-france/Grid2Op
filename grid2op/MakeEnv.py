# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

"""
The function define in this module is the easiest and most convenient ways to create a valid
:class:`grid2op.Environment.Environment`.

To get started with such an environment, you can simply do:

..code-block:: python

    import grid2op
    env = grid2op.make()


You can consult the different notebooks in the `getting_stared` directory of this package for more information on
how to use it.

Created Environment should behave exactly like a gym environment. If you notice any unwanted behavior, please address
an issue in the official grid2op repository: `Grid2Op <https://github.com/rte-france/Grid2Op>`_

The environment created with this method should be fully compatible with the gym framework: if you are developing
a new algorithm of "Reinforcement Learning" and you used the openai gym framework to do so, you can port your code
in a few minutes (basically this consists in adapting the input and output dimension of your BaseAgent) and make it work
with a Grid2Op environment. An example of such modifications is exposed in the getting_started/ notebooks.

"""
import os
import importlib.util
import pkg_resources
import warnings
import numbers
import pdb
import json

from grid2op.Environment import Environment
from grid2op.Backend import Backend, PandaPowerBackend
from grid2op.Parameters import Parameters
from grid2op.Chronics import ChronicsHandler, Multifolder, ChangeNothing
from grid2op.Chronics import GridStateFromFile, GridStateFromFileWithForecasts, GridValue
from grid2op.Action import BaseAction, TopologyAction, TopologyAndDispatchAction, DontAct
from grid2op.Exceptions import *
from grid2op.Observation import CompleteObservation, BaseObservation
from grid2op.Reward import FlatReward, BaseReward, L2RPNReward, RedispReward
from grid2op.Rules import BaseRules, AlwaysLegal, DefaultRules
from grid2op.VoltageControler import ControlVoltageFromFile
from grid2op.Opponent import BaseOpponent

from grid2op.Chronics.Settings_L2RPN2019 import L2RPN2019_CASEFILE, L2RPN2019_DICT_NAMES, ReadPypowNetData, CASE_14_L2RPN2019_LAYOUT
from grid2op.Chronics.Settings_5busExample import EXAMPLE_CHRONICSPATH, EXAMPLE_CASEFILE, CASE_5_GRAPH_LAYOUT
from grid2op.Chronics.Settings_case14_test import case14_test_CASEFILE, case14_test_CHRONICSPATH, case14_test_TH_LIM
from grid2op.Chronics.Settings_case14_redisp import case14_redisp_CASEFILE, case14_redisp_CHRONICSPATH, case14_redisp_TH_LIM
from grid2op.Chronics.Settings_case14_realistic import case14_real_CASEFILE, case14_real_CHRONICSPATH, case14_real_TH_LIM

CASE_14_FILE = os.path.abspath(os.path.join(pkg_resources.resource_filename(__name__, "data"),
                                            "test_PandaPower", "test_case14.json"))
CHRONICS_FODLER = os.path.abspath(os.path.join(pkg_resources.resource_filename(__name__, "data")))
CHRONICS_MLUTIEPISODE = os.path.join(CHRONICS_FODLER, "test_multi_chronics")

NAMES_CHRONICS_TO_BACKEND = {"loads": {"2_C-10.61": 'load_1_0', "3_C151.15": 'load_2_1',
                                       "14_C63.6": 'load_13_2', "4_C-9.47": 'load_3_3',
                                       "5_C201.84": 'load_4_4',
                                       "6_C-6.27": 'load_5_5', "9_C130.49": 'load_8_6',
                                       "10_C228.66": 'load_9_7',
                                       "11_C-138.89": 'load_10_8', "12_C-27.88": 'load_11_9',
                                       "13_C-13.33": 'load_12_10'},
                             "lines": {'1_2_1': '0_1_0', '1_5_2': '0_4_1', '9_10_16': '8_9_2',
                                       '9_14_17': '8_13_3',
                                       '10_11_18': '9_10_4', '12_13_19': '11_12_5', '13_14_20': '12_13_6',
                                       '2_3_3': '1_2_7', '2_4_4': '1_3_8', '2_5_5': '1_4_9',
                                       '3_4_6': '2_3_10',
                                       '4_5_7': '3_4_11', '6_11_11': '5_10_12', '6_12_12': '5_11_13',
                                       '6_13_13': '5_12_14', '4_7_8': '3_6_15', '4_9_9': '3_8_16',
                                       '5_6_10': '4_5_17',
                                       '7_8_14': '6_7_18', '7_9_15': '6_8_19'},
                             "prods": {"1_G137.1": 'gen_0_4', "3_G36.31": "gen_2_1", "6_G63.29": "gen_5_2",
                                       "2_G-56.47": "gen_1_0", "8_G40.43": "gen_7_3"},
                             }

ALLOWED_KWARGS_MAKE = {"param", "backend", "observation_class", "gamerules_class", "chronics_path", "reward_class",
                       "action_class", "grid_path", "names_chronics_to_backend", "data_feeding_kwargs",
                       "chronics_class", "volagecontroler_class", "other_rewards",
                       'opponent_action_class', "opponent_class", "opponent_init_budget"}
ALLOWED_KWARGS_MAKE2 = {"param", "backend", "observation_class", "gamerules_class", "reward_class",
                        "action_class", "data_feeding_kwargs", "chronics_class", "volagecontroler_class",
                        "other_rewards",
                       'opponent_action_class', "opponent_class", "opponent_init_budget"}
ERR_MSG_KWARGS = {
    "backend": "The backend of the environment (keyword \"backend\") must be an instance of grid2op.Backend",
    "observation": "The type of observation of the environment (keyword \"observation_class\")" \
    " must be a subclass of grid2op.BaseObservation",
    "param": "The parameters of the environment (keyword \"param\") must be an instance of grid2op.Parameters",
    "rules": "The type of rules of the environment (keyword \"gamerules_class\")" \
    " must be a subclass of grid2op.BaseRules",
    "reward": "The type of reward in the environment (keyword \"reward_class\") must be a subclass of grid2op.BaseReward",
    "action": "The type of action of the environment (keyword \"action_class\") must be a subclass of grid2op.BaseAction",
    "data_feeding_kwargs": "The argument to build the data generation process [chronics]" \
    "  (keyword \"data_feeding_kwargs\") should be a dictionnary.",
    "chronics": "The argument to build the data generation process [chronics] (keyword \"chronics_class\")" \
    " should be a class that inherit grid2op.Chronics.GridValue.",
    "chronics_handler": "The argument to build the data generation process [chronics] (keyword \"data_feeding\")" \
    " should be a class that inherit grid2op.ChronicsHandler.ChronicsHandler.",
    "voltage": "The argument to build the online controler for chronics (keyword \"volagecontroler_class\")" \
    " should be a class that inherit grid2op.VoltageControler.ControlVoltageFromFile.",
    "names_chronics_to_grid": "The converter between names (keyword \"names_chronics_to_backend\") should be a dictionnary.",
    "other_rewards": "The argument to build the online controler for chronics (keyword \"other_rewards\") "
                     "should be dictionnary.",
    "opponent_action_class": "The argument used to build the \"opponent_action_class\" should be a class that "
                             "inherit from \"BaseAction\"",
    "opponent_class": "The argument used to build the \"opponent_class\" should be a class that "
                             "inherit from \"BaseOpponent\"",
    "opponent_init_budget": "The initial budget of the opponent \"opponent_init_budget\" should be a float"
}
NAME_CHRONICS_FOLDER = "chronics"
NAME_GRID_FILE = "grid.json"
NAME_GRID_LAYOUT_FILE = "grid_layout.json"
NAME_CONFIG_FILE = "config.py"
# TODO read the "ALLOWED_KWARGS" from the key of ERR_MSG


def _get_default_aux(name, kwargs, defaultClassApp, _sentinel=None,
                     msg_error="Error when building the default parameter",
                     defaultinstance=None, defaultClass=None, build_kwargs={},
                     isclass=False):
    """
    Helper to build default parameters forwarded to :class:`grid2op.Environment.Environment` for its creation.

    Exactly one of ``defaultinstance`` or ``defaultClass`` should be used, and set to not ``None``

    Parameters
    ----------
    name: ``str``
        Name of the argument to look for

    kwargs: ``dict``
        The key word arguments given to the :func:`make` function

    defaultClassApp; ``type``
        The default class to which the returned object should belong to. The final object should either be an instance
        of this ``defaultClassApp`` (if isclass is ``False``) or a subclass of this (if isclass is ``True``)

    _sentinel: ``None``
        Internal, do not use. Present to force key word arguments.

    msg_error: ``str`` or ``None``
        The message error to display if the object does not belong to ``defaultClassApp``

    defaultinstance: ``object`` or ``None``
        The default instance that will be returned. Note that if ``defaultinstance`` is not None, then
        ``defaultClass`` should be ``None`` and ``build_kwargs`` and empty dictionnary.

    defaultClass: ``type`` or ``None``
        The class used to build the default object. Note that if ``defaultClass`` is not None, then
        ``defaultinstance`` should be.

    build_kwargs:  ``dict``
        The keyword arguments used to build the final object (if ``isclass`` is ``True``). Note that:

          * if ``isclass`` is ``False``, this should be empty
          * if ``defaultinstance`` is not None, then this should be empty
          * This parameter should allow to create a valid object of type ``defaultClass``: it's key must be
            proper keys accepted by the class

    isclass: ``bool``
        Whether to build an instance of a class, or just return the class.


    Returns
    -------
    res:
        The parameters, either read from kwargs, or with its default value.

    """
    err_msg = "Impossible to create the parameter \"{}\": "
    if _sentinel is not None:
        err_msg += "Impossible to get default parameters for building the environment. Please use keywords arguments."
        raise RuntimeError(err_msg)

    res = None
    # first seek for the parameter in the kwargs, and check it's valid
    if name in kwargs:
        res = kwargs[name]
        if isclass is False:
            # i must create an instance of a class. I check whether it's a instance.
            if not isinstance(res, defaultClassApp):
                if issubclass(defaultClassApp, numbers.Number):
                    try:
                        # if this is base numeric type, like float or anything, i try to convert to it (i want to
                        # accept that "int" are float for example.
                        res = defaultClassApp(res)
                    except:
                        # if there is any error, i raise the error message
                        raise EnvError(msg_error)
                else:
                    # if there is any error, i raise the error message
                    raise EnvError(msg_error)
        else:
            if not isinstance(res, type):
                raise EnvError("Parameter \"{}\" should be a type and not an instance. It means that you provided an "
                               "object instead of the class to build it.".format(name))
            # I must create a class, i check whether it's a subclass
            if not issubclass(res, defaultClassApp):
                raise EnvError(msg_error)

    if res is None:
        # build the default parameter if not found

        if isclass is False:
            # i need building an instance
            if defaultClass is not None:
                if defaultinstance is not None:
                    err_msg += "Impossible to build an environment with both a default instance, and a default class"
                    raise EnvError(err_msg.format(name))
                try:
                    res = defaultClass(**build_kwargs)
                except Exception as e:
                    e.args = e.args + ("Cannot create and instance of {} with parameters \"{}\"".format(defaultClass, build_kwargs),)
                    raise
            elif defaultinstance is not None:
                if len(build_kwargs):
                    err_msg += "An instance is provided, yet kwargs to build it is also provided"
                    raise EnvError(err_msg.format(name))
                res = defaultinstance
            else:
                err_msg = " None of \"defaultClass\" and \"defaultinstance\" is provided."
                raise EnvError(err_msg.format(name))
        else:
            # I returning a class
            if len(build_kwargs):
                err_msg += "A class must be returned, yet kwargs to build it is also provided"
                raise EnvError(err_msg.format(name))
            if defaultinstance is not None:
                err_msg += "A class must be returned yet a default instance is provided"
                raise EnvError(err_msg.format(name))
            res = defaultClass

    return res


def _check_kwargs(kwargs):
    for el in kwargs:
        if not el in ALLOWED_KWARGS_MAKE2:
            raise EnvError("Unknown keyword argument \"{}\" used to create an Environement. "
                           "No Environment will be created. "
                           "Accepted keyword arguments are {}".format(el, sorted(ALLOWED_KWARGS_MAKE2)))

def _check_path(path, info):
    if path is None or os.path.exists(path) is False:
        raise EnvError("Cannot find {}. {}".format(path, info))


def make2(dataset_path="/", **kwargs):
    """
    This function is a shortcut to rapidly create environments within the grid2op Framework.

    It mimic the ``gym.make`` function.

    Parameters
    ----------
    dataset_path: ``str``
        Path to the dataset folder

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

    Returns
    -------
    env: :class:`grid2op.Environment.Environment`
        The created environment.
    """
    # Compute and find root folder
    _check_path(dataset_path, "Dataset root directory")
    dataset_path_abs = os.path.abspath(dataset_path)
    # Compute env name from directory name
    name_env = os.path.split(dataset_path_abs)[1]
    
    # Compute and find chronics folder
    chronics_path_abs = os.path.abspath(os.path.join(dataset_path_abs, NAME_CHRONICS_FOLDER))
    _check_path(chronics_path_abs, "Dataset chronics folder")

    # Compute and find backend/grid file
    grid_path_abs = os.path.abspath(os.path.join(dataset_path_abs, NAME_GRID_FILE))
    _check_path(grid_path_abs, "Dataset power flow solver configuration")

    # Compute and find grid layout file
    grid_layout_path_abs = os.path.abspath(os.path.join(dataset_path_abs, NAME_GRID_LAYOUT_FILE))
    _check_path(grid_layout_path_abs, "Dataset grid layout")

    # Check provided config overrides are valid
    _check_kwargs(kwargs)

    # Compute and find config file
    config_path_abs = os.path.abspath(os.path.join(dataset_path_abs, NAME_CONFIG_FILE))
    _check_path(grid_path_abs, "Dataset environment configuration")
    # Read config file
    try:
        spec = importlib.util.spec_from_file_location("config.config", config_path_abs)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config_data = config_module.config
    except Exception as e:
        print (e)
        raise EnvError("Invalid dataset config file: {}".format(config_path_abs)) from None

    # Get graph layout
    try:
        with open(grid_layout_path_abs) as layout_fp:
            graph_layout = json.load(layout_fp)
    except Exception as e:
        raise EnvError("Dataset {} doesn't have a valid graph layout".format(config_path_abs))

    # Get thermal limits
    thermal_limits = None
    if "thermal_limits" in config_data:
        thermal_limits = config_data["thermal_limits"]

    # Get chronics_to_backend
    name_converter = None
    if "names_chronics_to_grid" in config_data:
        name_converter = config_data["names_chronics_to_grid"]
    if name_converter is None:
        name_converter = {}
    names_chronics_to_backend = _get_default_aux("names_chronics_to_backend", kwargs,
                                                 defaultClassApp=dict, defaultinstance=name_converter,
                                                 msg_error=ERR_MSG_KWARGS["names_chronics_to_grid"])
    # Get default backend class
    backend_class_cfg = PandaPowerBackend
    if "backend_class" in config_data and config_data["backend_class"] is not None:
        backend_class_cfg = config_data["backend_class"]
    ## Create the backend, to compute the powerflow
    backend = _get_default_aux("backend", kwargs, defaultClass=backend_class_cfg,
                               defaultClassApp=Backend, msg_error=ERR_MSG_KWARGS["backend"])

    # Get default observation class
    observation_class_cfg = CompleteObservation
    if "observation_class" in config_data and config_data["observation_class"] is not None:
        observation_class_cfg = config_data["observation_class"]
    ## Setup the type of observation the agent will receive
    observation_class = _get_default_aux("observation_class", kwargs, defaultClass=observation_class_cfg, isclass=True,
                                         defaultClassApp=BaseObservation, msg_error=ERR_MSG_KWARGS["observation"])
    
    ## Create the parameters of the game, thermal limits threshold,
    # simulate cascading failure, powerflow mode etc. (the gamification of the game)
    param = _get_default_aux('param', kwargs, defaultClass=Parameters,
                             defaultClassApp=Parameters, msg_error=ERR_MSG_KWARGS["param"])

    # Get default rules class
    rules_class_cfg = DefaultRules
    if "rules_class" in config_data and config_data["rules_class"] is not None:
        rules_class_cfg = config_data["rules_class"]
    ## Create the rules of the game (mimic the operationnal constraints)
    gamerules_class = _get_default_aux("gamerules_class", kwargs, defaultClass=rules_class_cfg,
                                       defaultClassApp=BaseRules, msg_error=ERR_MSG_KWARGS["rules"],
                                       isclass=True)

    # Get default reward class
    reward_class_cfg = L2RPNReward
    if "reward_class" in config_data and config_data["reward_class"] is not None:
        reward_class_cfg = config_data["reward_class"]
    ## Setup the reward the agent will receive
    reward_class = _get_default_aux("reward_class", kwargs, defaultClass=reward_class_cfg,
                                    defaultClassApp=BaseReward, msg_error=ERR_MSG_KWARGS["reward"],
                                    isclass=True)

    # Get default BaseAction class
    action_class_cfg = BaseAction
    if "action_class" in config_data and config_data["action_class"] is not None:
        action_class_cfg = config_data["action_class"]
    ## Setup the type of action the BaseAgent can perform
    action_class = _get_default_aux("action_class", kwargs, defaultClass=action_class_cfg,
                                    defaultClassApp=BaseAction, msg_error=ERR_MSG_KWARGS["action"],
                                    isclass=True)    
    
    # Get default Voltage class
    voltage_class_cfg = ControlVoltageFromFile
    if "voltage_class" in config_data and config_data["voltage_class"] is not None:
        voltage_class_cfg = config_data["voltage_class"]
    ### Create controler for voltages
    volagecontroler_class = _get_default_aux("volagecontroler_class", kwargs,
                                             defaultClassApp=voltage_class_cfg,
                                             defaultClass=ControlVoltageFromFile,
                                             msg_error=ERR_MSG_KWARGS["voltage"], isclass=True)

    # Get default Chronics class
    chronics_class_cfg = ChangeNothing
    if "chronics_class" in config_data and config_data["chronics_class"] is not None:
        chronics_class_cfg = config_data["chronics_class"]
    # Get default Grid class
    grid_value_class_cfg = GridStateFromFile
    if "grid_value_class" in config_data and config_data["grid_value_class"] is not None:
        grid_value_class_cfg = config_data["grid_value_class"]
    
    ## the chronics to use
    ### the arguments used to build the data, note that the arguments must be compatible with the chronics class
    default_chronics_kwargs = {
        "chronicsClass": chronics_class_cfg,
        "path": chronics_path_abs,
        "gridvalueClass": grid_value_class_cfg
    }

    data_feeding_kwargs = _get_default_aux("data_feeding_kwargs", kwargs,
                                           defaultClassApp=dict,
                                           defaultinstance=default_chronics_kwargs,
                                           msg_error=ERR_MSG_KWARGS["data_feeding_kwargs"])
    for el in default_chronics_kwargs:
        if not el in data_feeding_kwargs:
            data_feeding_kwargs[el] = default_chronics_kwargs[el]

    ### the chronics generator
    chronics_class_used = _get_default_aux("chronics_class", kwargs,
                                           defaultClassApp=GridValue,
                                           defaultClass=data_feeding_kwargs["chronicsClass"],
                                           msg_error=ERR_MSG_KWARGS["chronics"],
                                           isclass=True)
    data_feeding_kwargs["chronicsClass"] = chronics_class_used
    data_feeding = _get_default_aux("data_feeding", kwargs,
                                    defaultClassApp=ChronicsHandler,
                                    defaultClass=ChronicsHandler,
                                    build_kwargs=data_feeding_kwargs,
                                    msg_error=ERR_MSG_KWARGS["chronics_handler"])

    ### other rewards
    other_rewards = _get_default_aux("other_rewards", kwargs,
                                     defaultClassApp=dict,
                                     defaultinstance={},
                                     msg_error=ERR_MSG_KWARGS["other_rewards"],
                                     isclass=False)

    # Opponent
    # TODO make that in config file of the default environment !!!
    opponent_action_class = _get_default_aux("opponent_action_class",
                                             kwargs,
                                             defaultClassApp=BaseAction,
                                             defaultClass=DontAct,
                                             msg_error=ERR_MSG_KWARGS["opponent_action_class"],
                                             isclass=True)
    opponent_class = _get_default_aux("opponent_class",
                                      kwargs,
                                      defaultClassApp=BaseOpponent,
                                      defaultClass=BaseOpponent,
                                      msg_error=ERR_MSG_KWARGS["opponent_class"],
                                      isclass=True)
    opponent_init_budget = _get_default_aux("opponent_init_budget", kwargs,
                                            defaultClassApp=float,
                                            defaultinstance=0.,
                                            msg_error=ERR_MSG_KWARGS["opponent_init_budget"],
                                            isclass=False)

    # Finally instanciate env from config & overrides
    env = Environment(init_grid_path=grid_path_abs,
                      chronics_handler=data_feeding,
                      backend=backend,
                      parameters=param,
                      names_chronics_to_backend=names_chronics_to_backend,
                      actionClass=action_class,
                      observationClass=observation_class,
                      rewardClass=reward_class,
                      legalActClass=gamerules_class,
                      voltagecontrolerClass=volagecontroler_class,
                      other_rewards=other_rewards,
                      opponent_action_class=opponent_action_class,
                      opponent_class=opponent_class,
                      opponent_init_budget=opponent_init_budget)

    # Update the thermal limit if any
    if thermal_limits is not None:
        env.set_thermal_limit(thermal_limits)

    # Set graph layout
    env.attach_layout(graph_layout)

    return env
    
    
def make(name_env="case14_realistic", **kwargs):
    """
    This function is a shortcut to rapidly create some (pre defined) environments within the grid2op Framework.

    For now, only the environment corresponding to the IEEE "case14" powergrid, with some pre defined chronics
    is available.

    Other environments, with different powergrids will be made available in the future.

    It mimic the ``gym.make`` function.

    Parameters
    ----------
    name_env: ``str``
        Name of the environment to create.

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

    grid_path: ``str``, optional
        The path where the powergrid is located.
        If provided it must be a string, and point to a valid file present on the hard drive.

    data_feeding_kwargs: ``dict``, optional
        Dictionnary that is used to build the `data_feeding` (chronics) objects.

    chronics_class: ``type``, optional
        The type of chronics that represents the dynamics of the Environment created. Usually they come from different
        folders.

    data_feeding: ``type``, optional
        The type of chronics handler you want to use.

    chronics_path: ``str``
        Path where to look for the chronics dataset.

    volagecontroler_class: ``type``, optional
        The type of :class:`grid2op.VoltageControler.VoltageControler` to use, it defaults to

    other_rewards: ``dict``, optional
        Dictionnary with other rewards we might want to look at at during training. It is given as a dictionnary with
        keys the name of the reward, and the values a class representing the new variables.

    Returns
    -------
    env: :class:`grid2op.Environment.Environment`
        The created environment.
    """

    for el in kwargs:
        if not el in ALLOWED_KWARGS_MAKE:
            raise EnvError("Unknown keyword argument \"{}\" used to create an Environement. "
                           "No Environment will be created. "
                           "Accepted keyword arguments are {}".format(el, sorted(ALLOWED_KWARGS_MAKE)))

    # first extract parameters that doesn't not depend on the powergrid

    ## the parameters of the game, thermal limits threshold, simulate cascading failure, powerflow mode etc. (the gamification of the game)
    msg_error = "The parameters of the environment (keyword \"param\") must be an instance of grid2op.Parameters"
    param = _get_default_aux('param', kwargs, defaultClass=Parameters, defaultClassApp=Parameters,
                             msg_error=msg_error)

    ## the backend use, to compute the powerflow
    msg_error = "The backend of the environment (keyword \"backend\") must be an instance of grid2op.Backend"
    backend = _get_default_aux("backend", kwargs, defaultClass=PandaPowerBackend,
                               defaultClassApp=Backend,
                               msg_error=msg_error)

    ## type of observation the agent will receive
    msg_error = "The type of observation of the environment (keyword \"observation_class\")"
    msg_error += " must be a subclass of grid2op.BaseObservation"
    observation_class = _get_default_aux("observation_class", kwargs, defaultClass=CompleteObservation,
                                         defaultClassApp=BaseObservation,
                                         msg_error=msg_error,
                                         isclass=True)

    ## type of rules of the game (mimic the operationnal constraints)
    msg_error = "The path where the data is located (keyword \"chronics_path\") should be a string."
    chronics_path = _get_default_aux("chronics_path", kwargs,
                                     defaultClassApp=str, defaultinstance='',
                                     msg_error=msg_error)

    # bulid the default parameters for each case file
    data_feeding_default_class = ChronicsHandler
    gamerules_class = AlwaysLegal
    defaultinstance_chronics_kwargs = {}
    if name_env.lower() == "case14_fromfile":
        default_grid_path = CASE_14_FILE
        if chronics_path == '':
            chronics_path = CHRONICS_MLUTIEPISODE

        defaultinstance_chronics_kwargs = {"chronicsClass": Multifolder, "path": chronics_path,
                                           "gridvalueClass": GridStateFromFileWithForecasts}
        default_name_converter = NAMES_CHRONICS_TO_BACKEND
        default_action_class = TopologyAction
        default_reward_class = L2RPNReward
    elif name_env.lower() == "l2rpn_2019":
        if chronics_path == '':
            msg_error = "Default chronics (provided in this package) cannot be used with the environment "
            msg_error += "\"l2rpn_2019\". Please download the training data using either the method described in" \
                         "Grid2Op/l2rpn_2019/README.md (if you downloaded the github repository) or\n" \
                         "running the command line script (in a terminal):\n" \
                         "python -m grid2op.download --name \"l2rpn_2019\" --path_save PATH\WHERE\YOU\WANT\TO\DOWNLOAD"
            raise EnvError(msg_error)
        default_grid_path = L2RPN2019_CASEFILE
        defaultinstance_chronics_kwargs = {"chronicsClass": Multifolder, "path": chronics_path,
                                           "gridvalueClass": ReadPypowNetData}
        default_name_converter = L2RPN2019_DICT_NAMES
        default_action_class = TopologyAction
        default_reward_class = L2RPNReward
        gamerules_class = DefaultRules
    elif name_env.lower() == "case5_example":
        if chronics_path == '':
            chronics_path = EXAMPLE_CHRONICSPATH

        default_grid_path = EXAMPLE_CASEFILE
        defaultinstance_chronics_kwargs = {"chronicsClass": Multifolder, "path": chronics_path,
                                           "gridvalueClass": GridStateFromFileWithForecasts}
        default_name_converter = {}
        default_action_class = TopologyAction
        default_reward_class = L2RPNReward
        gamerules_class = DefaultRules
    elif name_env.lower() == "case14_test":
        if chronics_path == '':
            chronics_path = case14_test_CHRONICSPATH
            warnings.warn("Your are using a case designed for testing purpose. Consider using the \"case14_redisp\" "
                          "environment instead.")

        default_grid_path = case14_test_CASEFILE
        defaultinstance_chronics_kwargs = {"chronicsClass": Multifolder, "path": chronics_path,
                                           "gridvalueClass": GridStateFromFileWithForecasts}
        default_name_converter = {}
        default_action_class = TopologyAndDispatchAction
        default_reward_class = RedispReward
        gamerules_class = DefaultRules
    elif name_env.lower() == "case14_redisp":
        if chronics_path == '':
            chronics_path = case14_redisp_CHRONICSPATH
            warnings.warn("Your are using only 2 chronics for this environment. More can be download by running, "
                          "from a command line:\n"
                          "python -m grid2op.download --name \"case14_redisp\" "
                          "--path_save PATH\WHERE\YOU\WANT\TO\DOWNLOAD\DATA")

        default_grid_path = case14_redisp_CASEFILE
        defaultinstance_chronics_kwargs = {"chronicsClass": Multifolder, "path": chronics_path,
                                           "gridvalueClass": GridStateFromFileWithForecasts}
        default_name_converter = {}
        default_action_class = TopologyAndDispatchAction
        default_reward_class = RedispReward
        gamerules_class = DefaultRules
    elif name_env.lower() == "case14_realistic":
        if chronics_path == '':
            chronics_path = case14_real_CHRONICSPATH
            warnings.warn("Your are using only 2 chronics for this environment. More can be download by running, "
                          "from a command line:\n"
                          "python -m grid2op.download --name \"case14_realistic\" "
                          "--path_save PATH\WHERE\YOU\WANT\TO\DOWNLOAD\DATA")

        default_grid_path = case14_real_CASEFILE
        defaultinstance_chronics_kwargs = {"chronicsClass": Multifolder, "path": chronics_path,
                                           "gridvalueClass": GridStateFromFileWithForecasts}
        default_name_converter = {}
        default_action_class = TopologyAndDispatchAction
        default_reward_class = RedispReward
        gamerules_class = DefaultRules
    elif name_env.lower() == "blank":
        default_name_converter = {}
        default_grid_path = ""
        default_action_class = TopologyAction
        default_reward_class = L2RPNReward
        gamerules_class = AlwaysLegal
    else:
        raise UnknownEnv("Unknown Environment named \"{}\". Current known environments are \"case14_fromfile\" "
                         "(default), \"case5_example\", \"case14_redisp\", \"case14_realistic\" "
                         "and \"l2rpn_2019\"".format(name_env))

    if "chronicsClass" not in defaultinstance_chronics_kwargs:
        defaultinstance_chronics_kwargs["chronicsClass"] = ChangeNothing

    # extract powergrid dependant parameters
    ## type of rules of the game (mimic the operationnal constraints)
    msg_error = "The type of rules of the environment (keyword \"gamerules_class\")"
    msg_error += " must be a subclass of grid2op.BaseRules"
    gamerules_class = _get_default_aux("gamerules_class", kwargs, defaultClass=gamerules_class,
                                       defaultClassApp=BaseRules,
                                       msg_error=msg_error,
                                       isclass=True)

    ## type of reward the agent will receive
    msg_error = "The type of observation of the environment (keyword \"reward_class\")"
    msg_error += " must be a subclass of grid2op.BaseReward"
    reward_class = _get_default_aux("reward_class", kwargs, defaultClass=default_reward_class,
                                    defaultClassApp=BaseReward,
                                    msg_error=msg_error,
                                    isclass=True)

    ## type of action the BaseAgent can perform
    msg_error = "The type of action of the environment (keyword \"action_class\") must be a subclass of grid2op.BaseAction"
    action_class = _get_default_aux("action_class", kwargs, defaultClass=default_action_class,
                                    defaultClassApp=BaseAction,
                                    msg_error=msg_error,
                                    isclass=True)

    ## the powergrid path to use
    msg_error = "The path where the grid is located (keyword \"grid_path\") should be a string."
    grid_path = _get_default_aux("grid_path", kwargs,
                                 defaultClassApp=str, defaultinstance=default_grid_path,
                                 msg_error=msg_error)

    ##
    msg_error = "The converter between names (keyword \"names_chronics_to_backend\") should be a dictionnary."
    names_chronics_to_backend = _get_default_aux("names_chronics_to_backend", kwargs,
                                 defaultClassApp=dict, defaultinstance=default_name_converter,
                                 msg_error=msg_error)

    ## the chronics to use
    ### the arguments used to build the data, note that the arguments must be compatible with the chronics class
    msg_error = "The argument to build the data generation process [chronics] (keyword \"data_feeding_kwargs\")"
    msg_error += " should be a dictionnary."
    data_feeding_kwargs = _get_default_aux("data_feeding_kwargs", kwargs,
                                           defaultClassApp=dict,
                                           defaultinstance=defaultinstance_chronics_kwargs,
                                           msg_error=msg_error)
    for el in defaultinstance_chronics_kwargs:
        if not el in data_feeding_kwargs:
            data_feeding_kwargs[el] = defaultinstance_chronics_kwargs[el]

    ### the chronics generator
    msg_error = "The argument to build the data generation process [chronics] (keyword \"chronics_class\")"
    msg_error += " should be a class that inherit grid2op.ChronicsHandler.GridValue."
    chronics_class_used = _get_default_aux("chronics_class", kwargs,
                                           defaultClassApp=GridValue,
                                           defaultClass=data_feeding_kwargs["chronicsClass"],
                                           msg_error=msg_error,
                                           isclass=True)
    data_feeding_kwargs["chronicsClass"] = chronics_class_used

    ### the chronics generator
    msg_error = "The argument to build the data generation process [chronics] (keyword \"data_feeding\")"
    msg_error += " should be a class that inherit grid2op.ChronicsHandler.ChronicsHandler."
    data_feeding = _get_default_aux("data_feeding", kwargs,
                                    defaultClassApp=ChronicsHandler,
                                    defaultClass=data_feeding_default_class,
                                    build_kwargs=data_feeding_kwargs,
                                    msg_error=msg_error)

    ### controler for voltages
    msg_error = "The argument to build the online controler for chronics (keyword \"volagecontroler_class\")"
    msg_error += " should be a class that inherit grid2op.VoltageControler.ControlVoltageFromFile."
    volagecontroler_class = _get_default_aux("volagecontroler_class", kwargs,
                                             defaultClassApp=ControlVoltageFromFile,
                                             defaultClass=ControlVoltageFromFile,
                                             msg_error=msg_error,
                                             isclass=True)

    ### other rewards
    msg_error = "The argument to build the online controler for chronics (keyword \"other_rewards\")"
    msg_error += " should be dictionnary."
    other_rewards = _get_default_aux("other_rewards", kwargs,
                                     defaultClassApp=dict,
                                     defaultinstance={},
                                     msg_error=msg_error,
                                     isclass=False)

    # Opponent
    # TODO make that in config file of the default environment !!!
    opponent_action_class = _get_default_aux("opponent_action_class",
                                             kwargs,
                                             defaultClassApp=BaseAction,
                                             defaultClass=DontAct,
                                             msg_error=ERR_MSG_KWARGS["opponent_action_class"],
                                             isclass=True)
    opponent_class = _get_default_aux("opponent_class",
                                      kwargs,
                                      defaultClassApp=BaseOpponent,
                                      defaultClass=BaseOpponent,
                                      msg_error=ERR_MSG_KWARGS["opponent_class"],
                                      isclass=True)
    opponent_init_budget = _get_default_aux("opponent_init_budget", kwargs,
                                            defaultClassApp=float,
                                            defaultinstance=0.,
                                            msg_error=ERR_MSG_KWARGS["opponent_init_budget"],
                                            isclass=False)
    if not os.path.exists(grid_path):
        raise EnvError("There is noting at \"{}\" where the powergrid should be located".format(
            os.path.abspath(grid_path)))

    env = Environment(init_grid_path=grid_path,
                      chronics_handler=data_feeding,
                      backend=backend,
                      parameters=param,
                      names_chronics_to_backend=names_chronics_to_backend,
                      actionClass=action_class,
                      observationClass=observation_class,
                      rewardClass=reward_class,
                      legalActClass=gamerules_class,
                      voltagecontrolerClass=volagecontroler_class,
                      other_rewards=other_rewards,
                      opponent_action_class=opponent_action_class,
                      opponent_class=opponent_class,
                      opponent_init_budget=opponent_init_budget
                      )

    # update the thermal limit if any
    if name_env.lower() == "case14_test":
        env.set_thermal_limit(case14_test_TH_LIM)
        env.attach_layout(CASE_14_L2RPN2019_LAYOUT)
    elif name_env.lower() == "case14_redisp":
        env.set_thermal_limit(case14_redisp_TH_LIM)
        env.attach_layout(CASE_14_L2RPN2019_LAYOUT)
    elif name_env.lower() == "case14_realistic":
        env.set_thermal_limit(case14_real_TH_LIM)
        env.attach_layout(CASE_14_L2RPN2019_LAYOUT)
    elif name_env.lower() == "l2rpn_2019":
        env.attach_layout(CASE_14_L2RPN2019_LAYOUT)
    elif name_env.lower() == "case5_example":
        env.attach_layout(CASE_5_GRAPH_LAYOUT)
    return env
