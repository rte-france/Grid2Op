# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import importlib.util
import numpy as np
import json

from grid2op.Environment import Environment
from grid2op.Backend import Backend, PandaPowerBackend
from grid2op.Parameters import Parameters
from grid2op.Chronics import ChronicsHandler, ChangeNothing
from grid2op.Chronics import GridStateFromFile, GridValue
from grid2op.Action import BaseAction, DontAct
from grid2op.Exceptions import *
from grid2op.Observation import CompleteObservation, BaseObservation
from grid2op.Reward import BaseReward, L2RPNReward
from grid2op.Rules import BaseRules, DefaultRules
from grid2op.VoltageControler import ControlVoltageFromFile
from grid2op.Opponent import BaseOpponent, BaseActionBudget, NeverAttackBudget

from grid2op.MakeEnv.get_default_aux import _get_default_aux

DIFFICULTY_NAME = "difficulty"
CHALLENGE_NAME = "competition"
ERR_MSG_KWARGS = {
    "backend": "The backend of the environment (keyword \"backend\") must be an instance of grid2op.Backend",
    "observation_class": "The type of observation of the environment (keyword \"observation_class\")" \
    " must be a subclass of grid2op.BaseObservation",
    "param": "The parameters of the environment (keyword \"param\") must be an instance of grid2op.Parameters",
    "gamerules_class": "The type of rules of the environment (keyword \"gamerules_class\")" \
    " must be a subclass of grid2op.BaseRules",
    "reward_class": "The type of reward in the environment (keyword \"reward_class\") must be a subclass of grid2op.BaseReward",
    "action_class": "The type of action of the environment (keyword \"action_class\") must be a subclass of grid2op.BaseAction",
    "data_feeding_kwargs": "The argument to build the data generation process [chronics]" \
    "  (keyword \"data_feeding_kwargs\") should be a dictionnary.",
    "chronics_class": "The argument to build the data generation process [chronics] (keyword \"chronics_class\")" \
    " should be a class that inherit grid2op.Chronics.GridValue.",
    "chronics_handler": "The argument to build the data generation process [chronics] (keyword \"data_feeding\")" \
    " should be a class that inherit grid2op.ChronicsHandler.ChronicsHandler.",
    "voltagecontroler_class": "The argument to build the online controler for chronics (keyword \"volagecontroler_class\")" \
    " should be a class that inherit grid2op.VoltageControler.ControlVoltageFromFile.",
    "names_chronics_to_grid": "The converter between names (keyword \"names_chronics_to_backend\") should be a dictionnary.",
    "other_rewards": "The argument to build the online controler for chronics (keyword \"other_rewards\") "
                     "should be dictionnary.",

    "chronics_path": "The path where the data is located (keyword \"chronics_path\") should be a string.",
    "grid_path": "The path where the grid is located (keyword \"grid_path\") should be a string.",
    "opponent_action_class": "The argument used to build the \"opponent_action_class\" should be a class that "
                             "inherit from \"BaseAction\"",
    "opponent_class": "The argument used to build the \"opponent_class\" should be a class that "
                      "inherit from \"BaseOpponent\"",
    "opponent_attack_duration": "The number of time steps an attack from the opponent lasts",
    "opponent_attack_cooldown": "The number of time steps the opponent as to wait for an attack",
    "opponent_init_budget": "The initial budget of the opponent \"opponent_init_budget\" should be a float",
    "opponent_budget_class": "The opponent budget class (\"opponent_budget_class\") should derive from "
                             "\"BaseActionBudget\".",
    "opponent_budget_per_ts": "The increase of the opponent's budget (\"opponent_budget_per_ts\") should be a float.",
    "kwargs_opponent": "The extra kwargs argument used to properly initiliazed the opponent "
                       "(\"kwargs_opponent\") shoud "
                       "be a dictionary.",
    DIFFICULTY_NAME: "Unknown difficulty level {difficulty} for this environment. Authorized difficulties are "
                     "{difficulties}"
}

NAME_CHRONICS_FOLDER = "chronics"
NAME_GRID_FILE = "grid.json"
NAME_GRID_LAYOUT_FILE = "grid_layout.json"
NAME_CONFIG_FILE = "config.py"


def _check_kwargs(kwargs):
    for el in kwargs:
        if not el in ERR_MSG_KWARGS.keys():
            raise EnvError("Unknown keyword argument \"{}\" used to create an Environement. "
                           "No Environment will be created. "
                           "Accepted keyword arguments are {}".format(el, ERR_MSG_KWARGS.keys()))


def _check_path(path, info):
    if path is None or os.path.exists(path) is False:
        raise EnvError("Cannot find {}. {}".format(path, info))


def make_from_dataset_path(dataset_path="/", _add_to_name="", **kwargs):
    """
    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Prefer using the :func:`grid2op.make` function.

    This function is a shortcut to rapidly create environments within the grid2op Framework. We don't
    recommend using directly this function. Prefer using the :func:`make` function.

    It mimic the ``gym.make`` function.

    .. _Parameters-make-from-path:

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

    other_rewards: ``dict``, optional
        Used to additional information than the "info" returned value after a call to env.step.

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

    difficulty: ``str``, optional
        the difficulty level. If present it starts from "0" the "easiest" but least realistic mode. In the case of the
        dataset being used in the l2rpn competition, the level used for the competition is "competition" ("hardest" and
        most realistic mode). If multiple difficulty levels are available, the most realistic one
        (the "hardest") is the default choice.

    opponent_action_class: ``type``, optional
        The action class used for the opponent. The opponent will not be able to use action that are invalid with
        the given action class provided. It defaults to :class:`grid2op.Action.DontAct` which forbid any type
        of action possible.

    opponent_class: ``type``, optional
        The opponent class to use. The default class is :class:`grid2op.Opponent.BaseOpponent` which is a type
        of opponents that does nothing.

    opponent_init_budget: ``float``, optional
        The initial budget of the opponent. It defaults to 0.0 which means the opponent cannot perform any action
        if this is not modified.

    opponent_attack_duration: ``int``, optional
        The number of time steps an attack from the opponent lasts.

    opponent_attack_cooldown: ``int``, optional
        The number of time steps the opponent as to wait for an attack.

    opponent_budget_per_ts: ``float``, optional
        The increase of the opponent budget per time step. Each time step the opponent see its budget increase. It
        defaults to 0.0.

    opponent_budget_class: ``type``, optional
        defaults: :class:`grid2op.Opponent.UnlimitedBudget`

    _add_to_name:
        Internal, used for test only. Do not attempt to modify under any circumstances.

    Returns
    -------
    env: :class:`grid2op.Environment.Environment`
        The created environment with the given properties.

    """
    # Compute and find root folder
    _check_path(dataset_path, "Dataset root directory")
    dataset_path_abs = os.path.abspath(dataset_path)

    # Compute env name from directory name
    name_env = os.path.split(dataset_path_abs)[1]

    # Compute and find chronics folder
    chronics_path = _get_default_aux("chronics_path", kwargs,
                                     defaultClassApp=str, defaultinstance='',
                                     msg_error=ERR_MSG_KWARGS["chronics_path"])
    if chronics_path == "":
        # if no "chronics_path" argument is provided, look into the "chronics" folder
        chronics_path_abs = os.path.abspath(os.path.join(dataset_path_abs, NAME_CHRONICS_FOLDER))
    else:
        # otherwise use it
        chronics_path_abs = os.path.abspath(chronics_path)
    _check_path(chronics_path_abs, "Dataset chronics folder")

    # Compute and find backend/grid file
    grid_path = _get_default_aux("grid_path", kwargs,
                                 defaultClassApp=str, defaultinstance="",
                                 msg_error=ERR_MSG_KWARGS["grid_path"])
    if grid_path == "":
        grid_path_abs = os.path.abspath(os.path.join(dataset_path_abs, NAME_GRID_FILE))
    else:
        grid_path_abs = os.path.abspath(grid_path)
    _check_path(grid_path_abs, "Dataset power flow solver configuration")

    # Compute and find grid layout file
    grid_layout_path_abs = os.path.abspath(os.path.join(dataset_path_abs, NAME_GRID_LAYOUT_FILE))
    _check_path(grid_layout_path_abs, "Dataset grid layout")

    # Check provided config overrides are valid
    _check_kwargs(kwargs)

    # Compute and find config file
    config_path_abs = os.path.abspath(os.path.join(dataset_path_abs, NAME_CONFIG_FILE))
    _check_path(config_path_abs, "Dataset environment configuration")

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
                                         defaultClassApp=BaseObservation, msg_error=ERR_MSG_KWARGS["observation_class"])

    ## Create the parameters of the game, thermal limits threshold,
    # simulate cascading failure, powerflow mode etc. (the gamification of the game)
    if "param" in kwargs:
        param = _get_default_aux('param', kwargs, defaultClass=Parameters,
                                 defaultClassApp=Parameters, msg_error=ERR_MSG_KWARGS["param"])
    else:
        # param is not in kwargs
        param = Parameters()
        json_path = os.path.join(dataset_path_abs, "difficulty_levels.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                dict_ = json.load(f)
            available_parameters = sorted(dict_.keys())
            if DIFFICULTY_NAME in kwargs:
                # player enters a difficulty levels
                my_difficulty = kwargs[DIFFICULTY_NAME]
                try:
                    my_difficulty = str(my_difficulty)
                except:
                    raise EnvError("Impossible to convert your difficulty into a valid string. Please make sure to "
                                   "pass a string (eg \"2\") and not something else (eg. int(2)) as a difficulty")
                if my_difficulty in dict_:
                    param.init_from_dict(dict_[my_difficulty])
                else:
                    raise EnvError(ERR_MSG_KWARGS[DIFFICULTY_NAME].format(difficulty=my_difficulty,
                                                                          difficulties=available_parameters))
            else:
                # no difficulty name provided, i need to chose the most suited one
                if CHALLENGE_NAME in dict_:
                    param.init_from_dict(dict_[CHALLENGE_NAME])
                else:
                    # i chose the most difficult one
                    available_parameters_int = {}
                    for el in available_parameters:
                        try:
                            int_ = int(el)
                            available_parameters_int[int_] = el
                        except:
                            pass
                    max_ = np.max(list(available_parameters_int.keys()))
                    keys_ = available_parameters_int[max_]
                    param.init_from_dict(dict_[keys_])
        else:
            json_path = os.path.join(dataset_path_abs, "parameters.json")
            if os.path.exists(json_path):
                param.init_from_json(json_path)

    # Get default rules class
    rules_class_cfg = DefaultRules
    if "rules_class" in config_data and config_data["rules_class"] is not None:
        rules_class_cfg = config_data["rules_class"]
    ## Create the rules of the game (mimic the operationnal constraints)
    gamerules_class = _get_default_aux("gamerules_class", kwargs, defaultClass=rules_class_cfg,
                                       defaultClassApp=BaseRules, msg_error=ERR_MSG_KWARGS["gamerules_class"],
                                       isclass=True)

    # Get default reward class
    reward_class_cfg = L2RPNReward
    if "reward_class" in config_data and config_data["reward_class"] is not None:
        reward_class_cfg = config_data["reward_class"]
    ## Setup the reward the agent will receive
    reward_class = _get_default_aux("reward_class", kwargs, defaultClass=reward_class_cfg,
                                    defaultClassApp=BaseReward, msg_error=ERR_MSG_KWARGS["reward_class"],
                                    isclass=True)

    # Get default BaseAction class
    action_class_cfg = BaseAction
    if "action_class" in config_data and config_data["action_class"] is not None:
        action_class_cfg = config_data["action_class"]
    ## Setup the type of action the BaseAgent can perform
    action_class = _get_default_aux("action_class", kwargs, defaultClass=action_class_cfg,
                                    defaultClassApp=BaseAction, msg_error=ERR_MSG_KWARGS["action_class"],
                                    isclass=True)

    # Get default Voltage class
    voltage_class_cfg = ControlVoltageFromFile
    if "voltage_class" in config_data and config_data["voltage_class"] is not None:
        voltage_class_cfg = config_data["voltage_class"]
    ### Create controler for voltages
    volagecontroler_class = _get_default_aux("volagecontroler_class", kwargs,
                                             defaultClassApp=voltage_class_cfg,
                                             defaultClass=ControlVoltageFromFile,
                                             msg_error=ERR_MSG_KWARGS["voltagecontroler_class"], isclass=True)

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
                                           msg_error=ERR_MSG_KWARGS["chronics_class"],
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
    chronics_class_cfg = DontAct
    if "opponent_action_class" in config_data and config_data["opponent_action_class"] is not None:
        chronics_class_cfg = config_data["opponent_action_class"]
    opponent_action_class = _get_default_aux("opponent_action_class",
                                             kwargs,
                                             defaultClassApp=BaseAction,
                                             defaultClass=chronics_class_cfg,
                                             msg_error=ERR_MSG_KWARGS["opponent_action_class"],
                                             isclass=True)
    opponent_class_cfg = BaseOpponent
    if "opponent_class" in config_data and config_data["opponent_class"] is not None:
        opponent_class_cfg = config_data["opponent_class"]
    opponent_class = _get_default_aux("opponent_class",
                                      kwargs,
                                      defaultClassApp=BaseOpponent,
                                      defaultClass=opponent_class_cfg,
                                      msg_error=ERR_MSG_KWARGS["opponent_class"],
                                      isclass=True)
    opponent_budget_class_cfg = NeverAttackBudget
    if "opponent_budget_class" in config_data and config_data["opponent_budget_class"] is not None:
        opponent_budget_class_cfg = config_data["opponent_budget_class"]
    opponent_budget_class = _get_default_aux("opponent_budget_class",
                                             kwargs,
                                             defaultClassApp=BaseActionBudget,
                                             defaultClass=opponent_budget_class_cfg,
                                             msg_error=ERR_MSG_KWARGS["opponent_budget_class"],
                                             isclass=True)
    opponent_init_budget_cfg = 0.
    if "opponent_init_budget" in config_data and config_data["opponent_init_budget"] is not None:
        opponent_init_budget_cfg = config_data["opponent_init_budget"]
    opponent_init_budget = _get_default_aux("opponent_init_budget", kwargs,
                                            defaultClassApp=float,
                                            defaultinstance=opponent_init_budget_cfg,
                                            msg_error=ERR_MSG_KWARGS["opponent_init_budget"],
                                            isclass=False)
    opponent_budget_per_ts_cfg = 0.
    if "opponent_budget_per_ts" in config_data and config_data["opponent_budget_per_ts"] is not None:
        opponent_budget_per_ts_cfg = config_data["opponent_budget_per_ts"]
    opponent_budget_per_ts = _get_default_aux("opponent_budget_per_ts", kwargs,
                                              defaultClassApp=float,
                                              defaultinstance=opponent_budget_per_ts_cfg,
                                              msg_error=ERR_MSG_KWARGS["opponent_budget_per_ts"],
                                              isclass=False)
    opponent_attack_duration_cfg = 0
    if "opponent_attack_duration" in config_data and config_data["opponent_attack_duration"] is not None:
        opponent_attack_duration_cfg = config_data["opponent_attack_duration"]
    opponent_attack_duration = _get_default_aux("opponent_attack_duration", kwargs,
                                                defaultClassApp=int,
                                                defaultinstance=opponent_attack_duration_cfg,
                                                msg_error=ERR_MSG_KWARGS["opponent_attack_duration"],
                                                isclass=False)
    opponent_attack_cooldown_cfg = 99999
    if "opponent_attack_cooldown" in config_data and config_data["opponent_attack_cooldown"] is not None:
        opponent_attack_cooldown_cfg = config_data["opponent_attack_cooldown"]
    opponent_attack_cooldown = _get_default_aux("opponent_attack_cooldown", kwargs,
                                                defaultClassApp=int,
                                                defaultinstance=opponent_attack_cooldown_cfg,
                                                msg_error=ERR_MSG_KWARGS["opponent_attack_cooldown"],
                                                isclass=False)
    kwargs_opponent_cfg = {}
    if "kwargs_opponent" in config_data and config_data["kwargs_opponent"] is not None:
        kwargs_opponent_cfg = config_data["kwargs_opponent"]
    kwargs_opponent = _get_default_aux("kwargs_opponent", kwargs,
                                       defaultClassApp=dict,
                                       defaultinstance=kwargs_opponent_cfg,
                                       msg_error=ERR_MSG_KWARGS["kwargs_opponent"],
                                       isclass=False)

    # Finally instanciate env from config & overrides
    env = Environment(init_grid_path=grid_path_abs,
                      chronics_handler=data_feeding,
                      backend=backend,
                      parameters=param,
                      name=name_env+_add_to_name,
                      names_chronics_to_backend=names_chronics_to_backend,
                      actionClass=action_class,
                      observationClass=observation_class,
                      rewardClass=reward_class,
                      legalActClass=gamerules_class,
                      voltagecontrolerClass=volagecontroler_class,
                      other_rewards=other_rewards,
                      opponent_action_class=opponent_action_class,
                      opponent_class=opponent_class,
                      opponent_init_budget=opponent_init_budget,
                      opponent_attack_duration=opponent_attack_duration,
                      opponent_attack_cooldown=opponent_attack_cooldown,
                      opponent_budget_per_ts=opponent_budget_per_ts,
                      opponent_budget_class=opponent_budget_class,
                      kwargs_opponent=kwargs_opponent,
                      )

    # Update the thermal limit if any
    if thermal_limits is not None:
        env.set_thermal_limit(thermal_limits)

    # Set graph layout if not None and not an empty dict
    if graph_layout is not None and graph_layout:
        env.attach_layout(graph_layout)

    return env
