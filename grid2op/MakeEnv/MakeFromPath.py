# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import time
import importlib.util
import numpy as np
import json
import warnings

from grid2op.MakeEnv.PathUtils import USE_CLASS_IN_FILE
from grid2op.Environment import Environment
from grid2op.Backend import Backend, PandaPowerBackend
from grid2op.Opponent.opponentSpace import OpponentSpace
from grid2op.Parameters import Parameters
from grid2op.Chronics import (ChronicsHandler,
                              ChangeNothing,
                              FromNPY,
                              FromChronix2grid,
                              GridStateFromFile,
                              GridValue)
from grid2op.Action import BaseAction, DontAct
from grid2op.Exceptions import EnvError
from grid2op.Observation import CompleteObservation, BaseObservation
from grid2op.Reward import BaseReward, L2RPNReward
from grid2op.Rules import BaseRules, DefaultRules
from grid2op.VoltageControler import ControlVoltageFromFile
from grid2op.Opponent import BaseOpponent, BaseActionBudget, NeverAttackBudget
from grid2op.operator_attention import LinearAttentionBudget

from grid2op.MakeEnv.get_default_aux import _get_default_aux

DIFFICULTY_NAME = "difficulty"
CHALLENGE_NAME = "competition"
ERR_MSG_KWARGS = {
    "backend": 'The backend of the environment (keyword "backend") must be an instance of grid2op.Backend',
    "observation_class": 'The type of observation of the environment (keyword "observation_class")'
    " must be a subclass of grid2op.BaseObservation",
    "param": 'The parameters of the environment (keyword "param") must be an instance of grid2op.Parameters',
    "gamerules_class": 'The type of rules of the environment (keyword "gamerules_class")'
    " must be a subclass of grid2op.BaseRules",
    "reward_class": 'The type of reward in the environment (keyword "reward_class") must be a subclass of '
    "grid2op.BaseReward",
    "action_class": 'The type of action of the environment (keyword "action_class") must be a subclass of '
    "grid2op.BaseAction",
    "data_feeding_kwargs": "The argument to build the data generation process [chronics]"
    '  (keyword "data_feeding_kwargs") should be a dictionnary.',
    "chronics_class": 'The argument to build the data generation process [chronics] (keyword "chronics_class")'
    " should be a class that inherit grid2op.Chronics.GridValue.",
    "chronics_handler": 'The argument to build the data generation process [chronics] (keyword "data_feeding")'
    " should be a class that inherit grid2op.ChronicsHandler.ChronicsHandler.",
    "voltagecontroler_class": "The argument to build the online controler for chronics (keyword "
    '"volagecontroler_class")'
    " should be a class that inherit grid2op.VoltageControler.ControlVoltageFromFile.",
    "names_chronics_to_grid": 'The converter between names (keyword "names_chronics_to_backend") '
    "should be a dictionnary.",
    "other_rewards": 'The argument to build the online controler for chronics (keyword "other_rewards") '
    "should be dictionary.",
    "chronics_path": 'The path where the data is located (keyword "chronics_path") should be a string.',
    "grid_path": 'The path where the grid is located (keyword "grid_path") should be a string.',
    "opponent_space_type": 'The argument used to build the opponent space (expects a type / class and not an instance of that type)',
    "opponent_action_class": 'The argument used to build the "opponent_action_class" should be a class that '
    'inherit from "BaseAction"',
    "opponent_class": 'The argument used to build the "opponent_class" should be a class that '
    'inherit from "BaseOpponent"',
    "opponent_attack_duration": "The number of time steps an attack from the opponent lasts",
    "opponent_attack_cooldown": "The number of time steps the opponent as to wait for an attack",
    "opponent_init_budget": 'The initial budget of the opponent "opponent_init_budget" should be a float',
    "opponent_budget_class": 'The opponent budget class ("opponent_budget_class") should derive from '
    '"BaseActionBudget".',
    "opponent_budget_per_ts": 'The increase of the opponent\'s budget ("opponent_budget_per_ts") should be a float.',
    "kwargs_opponent": "The extra kwargs argument used to properly initialized the opponent "
    '("kwargs_opponent") should '
    "be a dictionary.",
    "has_attention_budget": 'The "has_attention_budget" key word argument should be a flag indicating whether '
    "you want this feature or not. It should be a boolean.",
    "attention_budget_class": 'The attention budget class ("attention_budget_class") should derive from '
    '"LinearAttentionBudget".',
    "kwargs_attention_budget": "The extra kwargs argument used to properly initialized the attention budget "
    '("kwargs_attention_budget") should '
    "be a dictionary.",
    DIFFICULTY_NAME: "Unknown difficulty level {difficulty} for this environment. Authorized difficulties are "
    "{difficulties}",
    "kwargs_observation": "The extra kwargs argument used to properly initialized each observations "
    '("kwargs_observation") should '
    "be a dictionary.",
    "observation_backend_class": ("The class used to build the observation backend (used for Simulator "
                                  "obs.simulate and obs.get_forecasted_env). If provided, this should "
                                  "be a type / class and not an instance of this class. (by default it's None)"),
    "observation_backend_kwargs": ("key-word arguments to build the observation backend (used for Simulator, "
    " obs.simulate and obs.get_forecasted_env). This should be a dictionnary. (by default it's None)")
}

NAME_CHRONICS_FOLDER = "chronics"
NAME_GRID_FILE = "grid"
NAME_GRID_LAYOUT_FILE = "grid_layout.json"
NAME_CONFIG_FILE = "config.py"


def _check_kwargs(kwargs):
    for el in kwargs:
        if el not in ERR_MSG_KWARGS.keys():
            raise EnvError(
                'Unknown keyword argument "{}" used to create an Environment. '
                "No Environment will be created. "
                "Accepted keyword arguments are {}".format(el, ERR_MSG_KWARGS.keys())
            )


def _check_path(path, info):
    if path is None or os.path.exists(path) is False:
        raise EnvError("Cannot find {}. {}".format(path, info))


def make_from_dataset_path(
    dataset_path="/",
    logger=None,
    experimental_read_from_local_dir=False,
    n_busbar=2,
    _add_to_name="",
    _compat_glop_version=None,
    **kwargs,
) -> Environment:
    """
    INTERNAL USE ONLY

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

    logger:
        Something to pass to grid2op environment to be used as logger.

    param: ``grid2op.Parameters.Parameters``, optional
        Type of parameters used for the Environment. Parameters defines how the powergrid problem is cast into an
        markov decision process, and some internal

    backend: ``grid2op.Backend.Backend``, optional
        The backend to use for the computation. If provided, it must be an instance of :class:`grid2op.Backend.Backend`.

    n_busbar: ``int``
        Number of independant busbars allowed per substations. By default it's 2.
        
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

    opponent_space_type: ``type``, optional
        The type of opponent space to use. If provided, it must be a subclass of `OpponentSpace`.
        
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

    kwargs_observation: ``dict``
        Key words used to initialize the observation. For example, in case of NoisyObservation, 
        it might be the standar error for each underlying distribution. It might
        be more complicated for other type of custom observations but should be
        deep copiable.

        Each observation will be initialized (by the observation_space) with:

        .. code-block:: python
        
            obs = observation_class(obs_env=self.obs_env,
                                    action_helper=self.action_helper_env,
                                    random_prng=self.space_prng,
                                    **kwargs_observation  # <- this kwargs is used here
                                   )

    observation_backend_class:
        The class used to build the observation backend (used for Simulator 
        obs.simulate and obs.get_forecasted_env). If provided, this should 
        be a type / class and not an instance of this class. (by default it's None)
        
    observation_backend_kwargs:
        The key-word arguments to build the observation backend (used for Simulator, 
        obs.simulate and obs.get_forecasted_env). This should be a dictionnary. 
        (by default it's None)
    
    _add_to_name:
        Internal, used for test only. Do not attempt to modify under any circumstances.

    _compat_glop_version:
        Internal, used for test only. Do not attempt to modify under any circumstances.

    # TODO update doc with attention budget

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
    chronics_path = _get_default_aux(
        "chronics_path",
        kwargs,
        defaultClassApp=str,
        defaultinstance="",
        msg_error=ERR_MSG_KWARGS["chronics_path"],
    )
    if chronics_path == "":
        # if no "chronics_path" argument is provided, look into the "chronics" folder
        chronics_path_abs = os.path.abspath(
            os.path.join(dataset_path_abs, NAME_CHRONICS_FOLDER)
        )
    else:
        # otherwise use it
        chronics_path_abs = os.path.abspath(chronics_path)
    exc_chronics = None
    try:
        _check_path(chronics_path_abs, "Dataset chronics folder")
    except Exception as exc_:
        exc_chronics = exc_

    # Compute and find grid layout file
    grid_layout_path_abs = os.path.abspath(
        os.path.join(dataset_path_abs, NAME_GRID_LAYOUT_FILE)
    )
    try:
        _check_path(grid_layout_path_abs, "Dataset grid layout")
    except EnvError as exc_:
        warnings.warn(
            f'Impossible to load the coordinate of the substation with error: "{exc_}". Expect some issue '
            f"if you attempt to plot the grid."
        )

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
    except Exception as exc_:
        print(exc_)
        raise EnvError(
            "Invalid dataset config file: {}".format(config_path_abs)
        ) from None

    # Get graph layout
    graph_layout = None
    try:
        with open(grid_layout_path_abs) as layout_fp:
            graph_layout = json.load(layout_fp)
    except Exception as exc_:
        warnings.warn(
            "Dataset {} doesn't have a valid graph layout. Expect some failures when attempting "
            "to plot the grid. Error was: {}".format(config_path_abs, exc_)
        )

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
        is_none = True
    else:
        is_none = False
    names_chronics_to_backend = _get_default_aux(
        "names_chronics_to_grid",
        kwargs,
        defaultClassApp=dict,
        defaultinstance=name_converter,
        msg_error=ERR_MSG_KWARGS["names_chronics_to_grid"],
    )
    if is_none and names_chronics_to_backend  == {}:
        names_chronics_to_backend = None
    
    # Get default backend class
    backend_class_cfg = PandaPowerBackend
    if "backend_class" in config_data and config_data["backend_class"] is not None:
        backend_class_cfg = config_data["backend_class"]
    ## Create the backend, to compute the powerflow
    backend = _get_default_aux(
        "backend",
        kwargs,
        defaultClass=backend_class_cfg,
        defaultClassApp=Backend,
        msg_error=ERR_MSG_KWARGS["backend"],
    )

    # Compute and find backend/grid file
    grid_path = _get_default_aux(
        "grid_path",
        kwargs,
        defaultClassApp=str,
        defaultinstance="",
        msg_error=ERR_MSG_KWARGS["grid_path"],
    )
    if grid_path == "":
        grid_path_abs = None
        for ext in backend.supported_grid_format:
            grid_path_abs = os.path.abspath(os.path.join(dataset_path_abs, f"{NAME_GRID_FILE}.{ext}"))
            try:
                _check_path(grid_path_abs, "Dataset power flow solver configuration")
                break
            except EnvError as exc_:
                pass
        if grid_path_abs is None:
            raise EnvError(f"Impossible to find a grid file format supported by your backend. Your backend said it supports "
                           f"the file with extension {backend.supported_grid_format}, "
                           f"none of which are found in '{dataset_path_abs}'")
    else:
        grid_path_abs = os.path.abspath(grid_path)
    _check_path(grid_path_abs, "Dataset power flow solver configuration")
    
    # Get default observation class
    observation_class_cfg = CompleteObservation
    if (
        "observation_class" in config_data
        and config_data["observation_class"] is not None
    ):
        observation_class_cfg = config_data["observation_class"]
    ## Setup the type of observation the agent will receive
    observation_class = _get_default_aux(
        "observation_class",
        kwargs,
        defaultClass=observation_class_cfg,
        isclass=True,
        defaultClassApp=BaseObservation,
        msg_error=ERR_MSG_KWARGS["observation_class"],
    )

    ## Create the parameters of the game, thermal limits threshold,
    # simulate cascading failure, powerflow mode etc. (the gamification of the game)
    if "param" in kwargs:
        param = _get_default_aux(
            "param",
            kwargs,
            defaultClass=Parameters,
            defaultClassApp=Parameters,
            msg_error=ERR_MSG_KWARGS["param"],
        )
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
                except Exception as exc_:
                    raise EnvError(
                        "Impossible to convert your difficulty into a valid string. Please make sure to "
                        'pass a string (eg "2") and not something else (eg. int(2)) as a difficulty.'
                        "Error was \n{}".format(exc_)
                    )
                if my_difficulty in dict_:
                    param.init_from_dict(dict_[my_difficulty])
                else:
                    raise EnvError(
                        ERR_MSG_KWARGS[DIFFICULTY_NAME].format(
                            difficulty=my_difficulty, difficulties=available_parameters
                        )
                    )
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
                        except Exception as exc_:
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
        warnings.warn("You used the deprecated rules_class in your config. Please change its "
                      "name to 'gamerules_class' to mimic the grid2op.make kwargs.")
        rules_class_cfg = config_data["rules_class"]
    if "gamerules_class" in config_data and config_data["gamerules_class"] is not None:
        rules_class_cfg = config_data["gamerules_class"]
        
    ## Create the rules of the game (mimic the operationnal constraints)
    gamerules_class = _get_default_aux(
        "gamerules_class",
        kwargs,
        defaultClass=rules_class_cfg,
        defaultClassApp=BaseRules,
        msg_error=ERR_MSG_KWARGS["gamerules_class"],
        isclass=None,
    )

    # Get default reward class
    reward_class_cfg = L2RPNReward
    if "reward_class" in config_data and config_data["reward_class"] is not None:
        reward_class_cfg = config_data["reward_class"]

    ## Setup the reward the agent will receive
    reward_class = _get_default_aux(
        "reward_class",
        kwargs,
        defaultClass=reward_class_cfg,
        defaultClassApp=BaseReward,
        msg_error=ERR_MSG_KWARGS["reward_class"],
        isclass=None,
    )

    # Get default BaseAction class
    action_class_cfg = BaseAction
    if "action_class" in config_data and config_data["action_class"] is not None:
        action_class_cfg = config_data["action_class"]
    ## Setup the type of action the BaseAgent can perform
    action_class = _get_default_aux(
        "action_class",
        kwargs,
        defaultClass=action_class_cfg,
        defaultClassApp=BaseAction,
        msg_error=ERR_MSG_KWARGS["action_class"],
        isclass=True,
    )

    # Get default Voltage class
    voltage_class_cfg = ControlVoltageFromFile
    if "voltage_class" in config_data and config_data["voltage_class"] is not None:
        voltage_class_cfg = config_data["voltage_class"]
    ### Create controler for voltages
    volagecontroler_class = _get_default_aux(
        "volagecontroler_class",
        kwargs,
        defaultClassApp=voltage_class_cfg,
        defaultClass=ControlVoltageFromFile,
        msg_error=ERR_MSG_KWARGS["voltagecontroler_class"],
        isclass=True,
    )

    # Get default Chronics class
    chronics_class_cfg = ChangeNothing
    if "chronics_class" in config_data and config_data["chronics_class"] is not None:
        chronics_class_cfg = config_data["chronics_class"]
        
    # Get default Grid class
    grid_value_class_cfg = GridStateFromFile
    if (
        "grid_value_class" in config_data
        and config_data["grid_value_class"] is not None
    ):
        grid_value_class_cfg = config_data["grid_value_class"]

    ## the chronics to use
    ### the arguments used to build the data, note that the arguments must be compatible with the chronics class
    default_chronics_kwargs = {
        "path": chronics_path_abs,
        "chronicsClass": chronics_class_cfg,
    }

    dfkwargs_cfg = {}  # in the config
    if "data_feeding_kwargs" in config_data and config_data["data_feeding_kwargs"] is not None:
        dfkwargs_cfg = config_data["data_feeding_kwargs"]
        for el in dfkwargs_cfg:
            default_chronics_kwargs[el] = dfkwargs_cfg[el]
            
    data_feeding_kwargs_user_prov = _get_default_aux(
        "data_feeding_kwargs",
        kwargs,
        defaultClassApp=dict,
        defaultinstance=default_chronics_kwargs,
        msg_error=ERR_MSG_KWARGS["data_feeding_kwargs"],
    )
    data_feeding_kwargs = data_feeding_kwargs_user_prov.copy()
    for el in default_chronics_kwargs:
        if el not in data_feeding_kwargs:
            data_feeding_kwargs[el] = default_chronics_kwargs[el]
            
            
    ### the chronics generator
    chronics_class_used = _get_default_aux(
        "chronics_class",
        kwargs,
        defaultClassApp=GridValue,
        defaultClass=data_feeding_kwargs["chronicsClass"],
        msg_error=ERR_MSG_KWARGS["chronics_class"],
        isclass=True,
    )
    if (
        ((chronics_class_used != ChangeNothing) and 
         (chronics_class_used != FromNPY) and 
         (chronics_class_used != FromChronix2grid))
    ) and exc_chronics is not None:
        raise EnvError(
            f"Impossible to find the chronics for your environment. Please make sure to provide "
            f'a folder "{NAME_CHRONICS_FOLDER}" within your environment folder.'
        )
    
    data_feeding_kwargs["chronicsClass"] = chronics_class_used
    if chronics_class_used.MULTI_CHRONICS:
        # add the default "gridvalueClass" in case of multi chronics and if the
        # parameters is not given in the "make" function but present in the config file
        if "gridvalueClass" not in data_feeding_kwargs:
            data_feeding_kwargs["gridvalueClass"] = grid_value_class_cfg
        
        
        # code bellow is added to fix
        # https://github.com/rte-france/Grid2Op/issues/593
        import inspect
        possible_params = inspect.signature(data_feeding_kwargs["gridvalueClass"].__init__).parameters
        data_feeding_kwargs_res = data_feeding_kwargs.copy()
        for el in data_feeding_kwargs:
            if el == "gridvalueClass":
                continue
            if el == "chronicsClass":
                continue
            if el not in possible_params:
                # if it's in the config but is not supported by the 
                # user, then we ignore it
                # see https://github.com/rte-france/Grid2Op/issues/593
                if el in dfkwargs_cfg and not el in data_feeding_kwargs_user_prov:
                    del data_feeding_kwargs_res[el]
        data_feeding_kwargs = data_feeding_kwargs_res
    # now build the chronics handler
    data_feeding = _get_default_aux(
        "data_feeding",
        kwargs,
        defaultClassApp=ChronicsHandler,
        defaultClass=ChronicsHandler,
        build_kwargs=data_feeding_kwargs,
        msg_error=ERR_MSG_KWARGS["chronics_handler"],
    )

    ### other rewards
    other_rewards_cfg = {}
    if "other_rewards" in config_data and config_data["other_rewards"] is not None:
        other_rewards_cfg = config_data["other_rewards"]
    other_rewards = _get_default_aux(
        "other_rewards",
        kwargs,
        defaultClassApp=dict,
        defaultinstance={},
        msg_error=ERR_MSG_KWARGS["other_rewards"],
        isclass=False,
    )
    for k in other_rewards_cfg:
        if k not in other_rewards:
            other_rewards[k] = other_rewards_cfg[k]

    # Opponent
    opponent_space_type_cfg = OpponentSpace
    if "opponent_space_type" in config_data and config_data["opponent_space_type"] is not None:
        opponent_space_type_cfg = config_data["opponent_space_type"]
    opponent_space_type = _get_default_aux(
        "opponent_space_type",
        kwargs,
        defaultClassApp=OpponentSpace,
        defaultClass=opponent_space_type_cfg,
        msg_error=ERR_MSG_KWARGS["opponent_space_type"],
        isclass=True,
    )
    
    chronics_class_cfg = DontAct
    if (
        "opponent_action_class" in config_data
        and config_data["opponent_action_class"] is not None
    ):
        chronics_class_cfg = config_data["opponent_action_class"]
    opponent_action_class = _get_default_aux(
        "opponent_action_class",
        kwargs,
        defaultClassApp=BaseAction,
        defaultClass=chronics_class_cfg,
        msg_error=ERR_MSG_KWARGS["opponent_action_class"],
        isclass=True,
    )
    opponent_class_cfg = BaseOpponent
    if "opponent_class" in config_data and config_data["opponent_class"] is not None:
        opponent_class_cfg = config_data["opponent_class"]
    opponent_class = _get_default_aux(
        "opponent_class",
        kwargs,
        defaultClassApp=BaseOpponent,
        defaultClass=opponent_class_cfg,
        msg_error=ERR_MSG_KWARGS["opponent_class"],
        isclass=True,
    )
    opponent_budget_class_cfg = NeverAttackBudget
    if (
        "opponent_budget_class" in config_data
        and config_data["opponent_budget_class"] is not None
    ):
        opponent_budget_class_cfg = config_data["opponent_budget_class"]
    opponent_budget_class = _get_default_aux(
        "opponent_budget_class",
        kwargs,
        defaultClassApp=BaseActionBudget,
        defaultClass=opponent_budget_class_cfg,
        msg_error=ERR_MSG_KWARGS["opponent_budget_class"],
        isclass=True,
    )
    opponent_init_budget_cfg = 0.0
    if (
        "opponent_init_budget" in config_data
        and config_data["opponent_init_budget"] is not None
    ):
        opponent_init_budget_cfg = config_data["opponent_init_budget"]
    opponent_init_budget = _get_default_aux(
        "opponent_init_budget",
        kwargs,
        defaultClassApp=float,
        defaultinstance=opponent_init_budget_cfg,
        msg_error=ERR_MSG_KWARGS["opponent_init_budget"],
        isclass=False,
    )
    opponent_budget_per_ts_cfg = 0.0
    if (
        "opponent_budget_per_ts" in config_data
        and config_data["opponent_budget_per_ts"] is not None
    ):
        opponent_budget_per_ts_cfg = config_data["opponent_budget_per_ts"]
    opponent_budget_per_ts = _get_default_aux(
        "opponent_budget_per_ts",
        kwargs,
        defaultClassApp=float,
        defaultinstance=opponent_budget_per_ts_cfg,
        msg_error=ERR_MSG_KWARGS["opponent_budget_per_ts"],
        isclass=False,
    )
    opponent_attack_duration_cfg = 0
    if (
        "opponent_attack_duration" in config_data
        and config_data["opponent_attack_duration"] is not None
    ):
        opponent_attack_duration_cfg = config_data["opponent_attack_duration"]
    opponent_attack_duration = _get_default_aux(
        "opponent_attack_duration",
        kwargs,
        defaultClassApp=int,
        defaultinstance=opponent_attack_duration_cfg,
        msg_error=ERR_MSG_KWARGS["opponent_attack_duration"],
        isclass=False,
    )
    opponent_attack_cooldown_cfg = 99999
    if (
        "opponent_attack_cooldown" in config_data
        and config_data["opponent_attack_cooldown"] is not None
    ):
        opponent_attack_cooldown_cfg = config_data["opponent_attack_cooldown"]
    opponent_attack_cooldown = _get_default_aux(
        "opponent_attack_cooldown",
        kwargs,
        defaultClassApp=int,
        defaultinstance=opponent_attack_cooldown_cfg,
        msg_error=ERR_MSG_KWARGS["opponent_attack_cooldown"],
        isclass=False,
    )
    kwargs_opponent_cfg = {}
    if "kwargs_opponent" in config_data and config_data["kwargs_opponent"] is not None:
        kwargs_opponent_cfg = config_data["kwargs_opponent"]
    kwargs_opponent = _get_default_aux(
        "kwargs_opponent",
        kwargs,
        defaultClassApp=dict,
        defaultinstance=kwargs_opponent_cfg,
        msg_error=ERR_MSG_KWARGS["kwargs_opponent"],
        isclass=False,
    )

    # attention budget
    has_attention_budget_cfg = False
    if (
        "has_attention_budget" in config_data
        and config_data["has_attention_budget"] is not None
    ):
        has_attention_budget_cfg = config_data["has_attention_budget"]
    has_attention_budget = _get_default_aux(
        "has_attention_budget",
        kwargs,
        defaultClassApp=bool,
        defaultinstance=has_attention_budget_cfg,
        msg_error=ERR_MSG_KWARGS["has_attention_budget"],
        isclass=False,
    )
    attention_budget_class_cfg = LinearAttentionBudget
    if (
        "attention_budget_class" in config_data
        and config_data["attention_budget_class"] is not None
    ):
        attention_budget_class_cfg = config_data["attention_budget_class"]
    attention_budget_class = _get_default_aux(
        "attention_budget_class",
        kwargs,
        defaultClassApp=LinearAttentionBudget,
        defaultClass=attention_budget_class_cfg,
        msg_error=ERR_MSG_KWARGS["attention_budget_class"],
        isclass=True,
    )

    kwargs_attention_budget_cfg = {}
    if (
        "kwargs_attention_budget" in config_data
        and config_data["kwargs_attention_budget"] is not None
    ):
        kwargs_attention_budget_cfg = config_data["kwargs_attention_budget"]
    kwargs_attention_budget = _get_default_aux(
        "kwargs_attention_budget",
        kwargs,
        defaultClassApp=dict,
        defaultinstance=kwargs_attention_budget_cfg,
        msg_error=ERR_MSG_KWARGS["kwargs_attention_budget"],
        isclass=False,
    )

    # observation key word arguments
    kwargs_observation = _get_default_aux(
        "kwargs_observation",
        kwargs,
        defaultClassApp=dict,
        defaultinstance={},
        msg_error=ERR_MSG_KWARGS["kwargs_observation"],
        isclass=False,
    )
    
    # backend for the observation 
    observation_backend_class_cfg = Backend
    if (
        "observation_backend_class" in config_data
        and config_data["observation_backend_class"] is not None
    ):
        observation_backend_class_cfg = config_data["observation_backend_class"]
    observation_backend_class = _get_default_aux(
        "observation_backend_class",
        kwargs,
        defaultClass=observation_backend_class_cfg,
        defaultClassApp=Backend,
        msg_error=ERR_MSG_KWARGS["observation_backend_class"],
        isclass=True,
    )
    if observation_backend_class is Backend:
        # in this case nothing is provided neither in the call to "make" 
        # nor in the config
        observation_backend_class = None
    
    # kwargs for observation backend
    observation_backend_kwargs_cfg_ = {"null": True} 
    # None and {} have specific meanings, so I "hack" it
    # to make the difference between "observation_backend_kwargs is not in config nor in 
    # the kwargs" and "observation_backend_kwargs is {} in the config or in the kwargs"
    observation_backend_kwargs_cfg = observation_backend_kwargs_cfg_
    if (
        "observation_backend_kwargs" in config_data
        and config_data["observation_backend_kwargs"] is not None
    ):
        observation_backend_kwargs_cfg = config_data["observation_backend_kwargs"]
    observation_backend_kwargs = _get_default_aux(
        "observation_backend_kwargs",
        kwargs,
        defaultClassApp=dict,
        defaultinstance=observation_backend_kwargs_cfg,
        msg_error=ERR_MSG_KWARGS["kwargs_observation"],
        isclass=False,
    ) 
    if observation_backend_kwargs is observation_backend_kwargs_cfg_:
        observation_backend_kwargs = None

    # new in 1.10.2 :
    allow_loaded_backend = False
    classes_path = None
    if USE_CLASS_IN_FILE:
        sys_path = os.path.join(os.path.split(grid_path_abs)[0], "_grid2op_classes")
        if not os.path.exists(sys_path):
            try:
                os.mkdir(sys_path)
            except FileExistsError:
                # if another process created it, no problem
                pass
            
        # TODO: automatic delete the directory if needed
        
        # TODO: check the "new" path works
        
        # TODO: in the BaseEnv.generate_classes make sure the classes are added to the "__init__" if the file is created
        # TODO: make that only if backend can be copied !
        
        # TODO: check the hash thingy is working in baseEnv._aux_gen_classes (currently a pdb)
        
        # TODO: check that previous behaviour is working correctly
        
        # TODO: create again the environment with the proper "read from local_dir"
        
        # TODO check that it works if the backend changes, if shunt / no_shunt if name of env changes etc.
        
        # TODO: what if it cannot write on disk => fallback to previous behaviour
        
        # TODO: allow for a way to disable that (with env variable or config in grid2op)
        # TODO: keep only one environment that will delete the files (with a flag in its constructor)
        
        # TODO: explain in doc new behaviour with regards to "class in file"
        
        # TODO: basic CI for this "new" mode
        
        # TODO: use the tempfile.TemporaryDirectory() to hold the classes, and in the (real) env copy, runner , env.get_kwargs() 
        # or whatever
        # reference this "tempfile.TemporaryDirectory()" which will be deleted automatically
        # when every "pointer" to it are deleted, this sounds more reasonable
        if not experimental_read_from_local_dir:
            init_env = Environment(init_env_path=os.path.abspath(dataset_path),
                                init_grid_path=grid_path_abs,
                                chronics_handler=data_feeding,
                                backend=backend,
                                parameters=param,
                                name=name_env + _add_to_name,
                                names_chronics_to_backend=names_chronics_to_backend,
                                actionClass=action_class,
                                observationClass=observation_class,
                                rewardClass=reward_class,
                                legalActClass=gamerules_class,
                                voltagecontrolerClass=volagecontroler_class,
                                other_rewards=other_rewards,
                                opponent_space_type=opponent_space_type,
                                opponent_action_class=opponent_action_class,
                                opponent_class=opponent_class,
                                opponent_init_budget=opponent_init_budget,
                                opponent_attack_duration=opponent_attack_duration,
                                opponent_attack_cooldown=opponent_attack_cooldown,
                                opponent_budget_per_ts=opponent_budget_per_ts,
                                opponent_budget_class=opponent_budget_class,
                                kwargs_opponent=kwargs_opponent,
                                has_attention_budget=has_attention_budget,
                                attention_budget_cls=attention_budget_class,
                                kwargs_attention_budget=kwargs_attention_budget,
                                logger=logger,
                                n_busbar=n_busbar,  # TODO n_busbar_per_sub different num per substations: read from a config file maybe (if not provided by the user)
                                _compat_glop_version=_compat_glop_version,
                                _read_from_local_dir=None,  # first environment to generate the classes and save them
                                kwargs_observation=kwargs_observation,
                                observation_bk_class=observation_backend_class,
                                observation_bk_kwargs=observation_backend_kwargs,
                                )              
            this_local_dir = f"{time.time()}_{os.getpid()}"
            init_env.generate_classes(local_dir_id=this_local_dir)
            init_env.backend = None  # to avoid to close the backend when init_env is deleted
            classes_path = os.path.join(sys_path, this_local_dir)
            # to force the reading back of the classes from the hard drive
            init_env._forget_classes()  # TODO not implemented
            init_env.close()
        else:
            classes_path = sys_path
        allow_loaded_backend = True
    else:
        # legacy behaviour (<= 1.10.1 behaviour)
        classes_path = None if not experimental_read_from_local_dir else experimental_read_from_local_dir
        if experimental_read_from_local_dir:
            sys_path = os.path.join(os.path.split(grid_path_abs)[0], "_grid2op_classes")
            if not os.path.exists(sys_path):
                raise RuntimeError(
                    "Attempting to load the grid classes from the env path. Yet the directory "
                    "where they should be placed does not exists. Did you call `env.generate_classes()` "
                    "BEFORE creating an environment with `experimental_read_from_local_dir=True` ?"
                )
            if not os.path.isdir(sys_path) or not os.path.exists(
                os.path.join(sys_path, "__init__.py")
            ):
                raise RuntimeError(
                    f"Impossible to load the classes from the env path. There is something that is "
                    f"not a directory and that is called `_grid2op_classes`. "
                    f'Please remove "{sys_path}" and call `env.generate_classes()` where env is an '
                    f"environment created with `experimental_read_from_local_dir=False` (default)"
                )
            
    # Finally instantiate env from config & overrides
    # including (if activated the new grid2op behaviour)
    env = Environment(
        init_env_path=os.path.abspath(dataset_path),
        init_grid_path=grid_path_abs,
        chronics_handler=data_feeding,
        backend=backend,
        parameters=param,
        name=name_env + _add_to_name,
        names_chronics_to_backend=names_chronics_to_backend,
        actionClass=action_class,
        observationClass=observation_class,
        rewardClass=reward_class,
        legalActClass=gamerules_class,
        voltagecontrolerClass=volagecontroler_class,
        other_rewards=other_rewards,
        opponent_space_type=opponent_space_type,
        opponent_action_class=opponent_action_class,
        opponent_class=opponent_class,
        opponent_init_budget=opponent_init_budget,
        opponent_attack_duration=opponent_attack_duration,
        opponent_attack_cooldown=opponent_attack_cooldown,
        opponent_budget_per_ts=opponent_budget_per_ts,
        opponent_budget_class=opponent_budget_class,
        kwargs_opponent=kwargs_opponent,
        has_attention_budget=has_attention_budget,
        attention_budget_cls=attention_budget_class,
        kwargs_attention_budget=kwargs_attention_budget,
        logger=logger,
        n_busbar=n_busbar,  # TODO n_busbar_per_sub different num per substations: read from a config file maybe (if not provided by the user)
        _compat_glop_version=_compat_glop_version,
        _read_from_local_dir=classes_path,
        _allow_loaded_backend=allow_loaded_backend,
        kwargs_observation=kwargs_observation,
        observation_bk_class=observation_backend_class,
        observation_bk_kwargs=observation_backend_kwargs,
    )
            
    # Update the thermal limit if any
    if thermal_limits is not None:
        env.set_thermal_limit(thermal_limits)

    # Set graph layout if not None and not an empty dict
    if graph_layout is not None and graph_layout:
        env.attach_layout(graph_layout)

    return env
