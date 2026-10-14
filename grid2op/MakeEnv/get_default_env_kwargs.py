# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import importlib
import os
from typing import Any, Dict, Literal, Tuple, Type, Union, Optional
import numpy as np
import json
import warnings


from grid2op.Backend import Backend, PandaPowerBackend
from grid2op.Opponent.opponentSpace import OpponentSpace
from grid2op.Parameters import Parameters
from grid2op.Chronics import (ChronicsHandler,
                              ChangeNothing,
                              FromNPY,
                              FromChronix2grid,
                              GridStateFromFile,
                              FromHandlers,
                              GridValue)
from grid2op.Action import BaseAction, DontAct
from grid2op.Exceptions import EnvError
from grid2op.Environment.dispatch import BaseRedispatchSolver
from grid2op.Observation import CompleteObservation, BaseObservation
from grid2op.Reward import BaseReward, L2RPNReward
from grid2op.Rules import BaseRules, DefaultRules
from grid2op.VoltageControler import ControlVoltageFromFile
from grid2op.Opponent import BaseOpponent, BaseActionBudget, NeverAttackBudget
from grid2op.operator_attention import LinearAttentionBudget
from grid2op.typing_variables import DICT_CONFIG_TYPING

from grid2op.MakeEnv.PathUtils import USE_CLASS_IN_FILE
from grid2op.MakeEnv.get_default_aux import _get_default_aux

try:
    from typing import Unpack, TypedDict
except ImportError:
    from typing_extensions import Unpack, TypedDict


class MakeKwargsTypeHints(TypedDict, total=False):
    backend: Backend
    observation_class: Type[BaseObservation]
    param: Union[Parameters, Dict[str, Any]]
    gamerules_class: Type[BaseRules]
    reward_class: Union[BaseReward, Type[BaseReward]]
    action_class: Type[BaseAction] 
    data_feeding_kwargs: Dict[str, Any]
    chronics_class: Type[GridValue]
    chronics_handler: ChronicsHandler
    voltagecontroler_class: Type[ControlVoltageFromFile]
    names_chronics_to_grid: Dict[Literal["loads", "prods", "lines", "subs"], Dict[str, str]]
    other_rewards: Dict[str, Union[BaseReward, Type[BaseReward]]]
    chronics_path: str
    grid_path: str
    opponent_space_type: Type[OpponentSpace]
    opponent_action_class: Type[BaseAction]
    opponent_class: Type[BaseOpponent]
    opponent_attack_duration: int
    opponent_attack_cooldown: int
    opponent_init_budget: float
    opponent_budget_class: Type[BaseActionBudget]
    opponent_budget_per_ts: float
    kwargs_opponent: Dict[str, Any]
    has_attention_budget: bool
    attention_budget_class: Type[LinearAttentionBudget]
    kwargs_attention_budget: Dict[str, Any]
    difficulty: str
    kwargs_observation: Dict[str, Any]
    observation_backend_class: Optional[Type[Backend]]
    observation_backend_kwargs: Optional[Dict[str, Any]]
    class_in_file: bool
    redispatch_solver: Union[BaseRedispatchSolver, Type[BaseRedispatchSolver]]
    
    
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
    " obs.simulate and obs.get_forecasted_env). This should be a dictionnary. (by default it's None)"),
    "class_in_file": ("experimental: tell grid2op to store the classes generated in the hard drive "
                      "which can solve lots of pickle / multi processing related issue"),
    "redispatch_solver": ("The solver used to compute the dispatch of the generators (keyword "
                          '"redispatch_solver") should be an instance or a subclass of '
                          '"grid2op.Environment.dispatch.BaseRedispatchSolver" (by default it\'s None '
                          "and the DefaultRedispatchSolver is used)"),
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
        
def get_default_env_kwargs(
    *,
    dataset_path,
    logger,
    n_busbar,
    allow_detachment,
    _add_cls_nm_bk,
    _add_to_name,
    _compat_glop_version,
    _overload_name_multimix,
    _warn_layout_missing=True,
    **kwargs: Unpack[MakeKwargsTypeHints]
    ):
    
    # full dataset path
    dataset_path_abs : str = os.path.abspath(dataset_path)
    
    # Compute env name from directory name
    name_env : str = os.path.split(dataset_path_abs)[1]

    # Compute and find grid layout file
    grid_layout_path_abs : str = os.path.abspath(
        os.path.join(dataset_path_abs, NAME_GRID_LAYOUT_FILE)
    )
    try:
        _check_path(grid_layout_path_abs, "Dataset grid layout")
    except EnvError as exc_:
        if _warn_layout_missing:
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
        config_data : DICT_CONFIG_TYPING = config_module.config
    except Exception as exc_:
        print(exc_)
        raise EnvError(
            f"Invalid dataset config file\n{exc_}: {config_path_abs}"
        ) from exc_


    default_ch_path = NAME_CHRONICS_FOLDER
    if "chronics_path" in config_data and config_data["chronics_path"] is not None:
        default_ch_path = config_data["chronics_path"]
    # Compute and find chronics folder
    chronics_path : str = _get_default_aux(
        "chronics_path",
        kwargs,
        defaultClassApp=str,
        defaultinstance=default_ch_path,
        msg_error=ERR_MSG_KWARGS["chronics_path"],
    )
    if chronics_path == NAME_CHRONICS_FOLDER:
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
        
    # Get graph layout
    graph_layout : Optional[Dict[str, Tuple[float, float]]] = None
    try:
        with open(grid_layout_path_abs) as layout_fp:
            graph_layout = json.load(layout_fp)
    except Exception as exc_:
        if _warn_layout_missing:
            warnings.warn(
                "Dataset {} doesn't have a valid graph layout. Expect some failures when attempting "
                "to plot the grid. Error was: {}".format(config_path_abs, exc_)
            )

    # Get thermal limits
    thermal_limits : Optional[Union[np.ndarray, Dict[str, float]]] = None
    if "thermal_limits" in config_data:
        thermal_limits = config_data["thermal_limits"]

    # Get chronics_to_backend
    name_converter = None
    if "names_chronics_to_grid" in config_data:
        name_converter = copy.deepcopy(config_data["names_chronics_to_grid"])
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
    backend_inst_cfg : Optional[Backend] = None
    backend_class_cfg : Optional[Type[Backend]]= PandaPowerBackend
    # new in 1.12.4 allow usage of backend directly in the config
    if "backend" in config_data and config_data["backend"] is not None:
        if isinstance(config_data["backend"], type):
            # legacy behaviour... default backend is given with a class
            # but named "backend"...
            backend_class_cfg = config_data["backend"]
        else:
            # user provided a valid backend instance
            backend_inst_cfg = config_data["backend"].copy()
            backend_class_cfg = None
    elif "backend_class" in config_data and config_data["backend_class"] is not None:
        backend_class_cfg= config_data["backend_class"]
        
    ## Create the backend, to compute the powerflow
    backend : Backend = _get_default_aux(
        "backend",
        kwargs,
        defaultinstance=backend_inst_cfg,
        defaultClass=backend_class_cfg,
        defaultClassApp=Backend,
        msg_error=ERR_MSG_KWARGS["backend"],
    )

    # Compute and find backend/grid file
    default_grid_path = ""
    if "grid_path" in config_data and config_data["grid_path"] is not None:
        default_grid_path = config_data["grid_path"]
    grid_path : str = _get_default_aux(
        "grid_path",
        kwargs,
        defaultClassApp=str,
        defaultinstance=default_grid_path,
        msg_error=ERR_MSG_KWARGS["grid_path"],
    )
    if grid_path == "":
        grid_path_abs = None
        for ext in backend.supported_grid_format:
            grid_path_abs = os.path.abspath(os.path.join(dataset_path_abs, f"{NAME_GRID_FILE}.{ext}"))
            try:
                _check_path(grid_path_abs, "Dataset power flow solver configuration")
                break
            except EnvError as exc_:  # noqa: F841
                grid_path_abs = None
                
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
        observation_class_cfg : Type[BaseObservation] = config_data["observation_class"]
    ## Setup the type of observation the agent will receive
    observation_class : Type[BaseObservation] = _get_default_aux(
        "observation_class",
        kwargs,
        defaultClass=observation_class_cfg,
        isclass=True,
        defaultClassApp=BaseObservation,
        msg_error=ERR_MSG_KWARGS["observation_class"],
    )

    ## Create the parameters of the game, thermal limits threshold,
    # simulate cascading failure, powerflow mode etc. (the gamification of the game)
    if "param" in config_data and config_data["param"] is not None:
        raise EnvError("If you want to change the default parameters, "
                       "please provide a 'parameters.json' file at the location of the dataset instead "
                       "relying on the 'config' key of the config.py")
    
    if "param" in kwargs:
        param : Parameters = _get_default_aux(
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
                        except ValueError as exc_:  # noqa: F841
                            # parameter key is not an int
                            # it is not considered
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
        rules_class_cfg : Type[BaseRules] = config_data["rules_class"]
    if "gamerules_class" in config_data and config_data["gamerules_class"] is not None:
        rules_class_cfg : Type[BaseRules] = config_data["gamerules_class"]
        
    ## Create the rules of the game (mimic the operationnal constraints)
    gamerules_class : Type[BaseRules] = _get_default_aux(
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
        reward_class_cfg : Type[BaseReward] = config_data["reward_class"]

    ## Setup the reward the agent will receive
    reward_class : Type[BaseReward] = _get_default_aux(
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
    if "voltagecontroler_class" in config_data and config_data["voltagecontroler_class"] is not None:
        voltage_class_cfg = config_data["voltagecontroler_class"]
    ### Create controler for voltages
    volagecontroler_class = _get_default_aux(
        "voltagecontroler_class",
        kwargs,
        defaultClassApp=ControlVoltageFromFile,
        defaultClass=voltage_class_cfg,
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
         (chronics_class_used != FromChronix2grid) and
         (chronics_class_used != FromHandlers)
         )
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
        # https://github.com/Grid2Op/grid2op/issues/593
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
                # see https://github.com/Grid2Op/grid2op/issues/593
                if el in dfkwargs_cfg and el not in data_feeding_kwargs_user_prov:
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

    # solver for the redispatching (new in 1.12.6)
    redispatch_solver = None
    if config_data.get("redispatch_solver") is not None:
        redispatch_solver = config_data["redispatch_solver"]
    if kwargs.get("redispatch_solver") is not None:
        redispatch_solver = kwargs["redispatch_solver"]
    if redispatch_solver is not None and not (
        isinstance(redispatch_solver, BaseRedispatchSolver)
        or (isinstance(redispatch_solver, type) and issubclass(redispatch_solver, BaseRedispatchSolver))
    ):
        raise EnvError(ERR_MSG_KWARGS["redispatch_solver"])

    # new in 1.10.2 :
    allow_loaded_backend = False
    classes_path = None
    init_env = None
    this_local_dir = None
    use_class_in_files = USE_CLASS_IN_FILE
    if "class_in_file" in kwargs:
        classes_in_file_kwargs = bool(kwargs["class_in_file"])
        use_class_in_files = classes_in_file_kwargs
        
    # new in 1.11.0:
    if _add_cls_nm_bk:
        _add_to_name = backend.get_class_added_name() + _add_to_name
    do_not_erase_cls : Optional[bool] = None
    
    # new in 1.11.0
    if _overload_name_multimix is not None:
        # this is a multimix
        # AND this is the first mix of a multi mix
        # I change the env name to add the "add_to_name"
        
        if  _overload_name_multimix.mix_id == 0:  
            # this is the first mix I need to assign proper names
            _overload_name_multimix.name_env = _overload_name_multimix.name_env + _add_to_name
            _overload_name_multimix.add_to_name = ""
        else:
            # this is not the first mix
            # for the other mix I need to read the data from files and NOT
            # create the classes
            use_class_in_files = False
            _add_to_name = ''  # already defined in the first mix
            name_env = _overload_name_multimix.name_env
    
    name_env = name_env + _add_to_name
    default_kwargs = dict(
        init_env_path=os.path.abspath(dataset_path),
        init_grid_path=grid_path_abs,
        backend=backend,
        parameters=param,
        name=name_env,
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
        _overload_name_multimix=_overload_name_multimix,
        kwargs_observation=kwargs_observation,
        observation_bk_class=observation_backend_class,
        observation_bk_kwargs=observation_backend_kwargs,
        allow_detachment=allow_detachment,
        redispatch_solver=redispatch_solver,
    )
    return (
        default_kwargs,
        use_class_in_files,
        grid_path_abs,
        data_feeding,
        graph_layout,
        backend,
        thermal_limits,
        allow_loaded_backend,
        classes_path,
        init_env,
        this_local_dir,
        do_not_erase_cls
    )

