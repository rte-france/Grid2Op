# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import os
import json
import warnings

from grid2op.dtypes import dt_int, dt_float, dt_bool


class Parameters:
    """
    Main classes representing the parameters of the game. The main parameters are described bellow.

    Note that changing the values of these parameters might not be enough. If these _parameters are not used in the
    :class:`grid2op.Rules.RulesChecker`, then modifying them will have no impact at all.

    Attributes
    ----------
    NO_OVERFLOW_DISCONNECTION: ``bool``
        If set to ``True`` then the :class:`grid2op.Environment.Environment` will not disconnect powerline above their
        thermal
        limit. Default is ``False``

    NB_TIMESTEP_OVERFLOW_ALLOWED: ``int``
        Number of timesteps for which a soft overflow is allowed, default 2. This means that a powerline will be
        disconnected (if :attr:`.NO_OVERFLOW_DISCONNECTION` is set to ``False``) after 2 time steps above its thermal
        limit. This is called a "soft overflow".

    NB_TIMESTEP_RECONNECTION: ``int``
        Number of timesteps a powerline disconnected for security motives (for example due to
        :attr:`.NB_TIMESTEP_POWERFLOW_ALLOWED` or :attr:`.HARD_OVERFLOW_THRESHOLD`) will remain disconnected.
        It's set to 10 timestep by default.

    NB_TIMESTEP_COOLDOWN_LINE: ``int``
        When someone acts on a powerline by changing its status (connected / disconnected) this number indicates
        how many timesteps the :class:`grid2op.Agent.BaseAgent` has to wait before being able to modify this status
        again.
        For examle, if this is 1, this means that an BaseAgent can act on status of a powerline 1 out of 2 time step (1
        time step it acts, another one it cools down, and the next one it can act again). Having it at 0 it equivalent
        to deactivate this feature (default).

    NB_TIMESTEP_COOLDOWN_SUB: ``int``
        When someone changes the topology of a substations, this number indicates how many timesteps the
        :class:`grid2op.Agent.BaseAgent` has to wait before being able to modify the topology on this same substation. It
        has the same behaviour as :attr:`Parameters.NB_TIMESTEP_LINE_STATUS_REMODIF`. To deactivate this feature,
        put it at 0 (default).

    HARD_OVERFLOW_THRESHOLD: ``float``
        If a the powerflow on a line is above HARD_OVERFLOW_THRESHOLD * thermal limit (and
        :attr:`Parameters.NO_OVERFLOW_DISCONNECTION` is set to ``False``) then it is automatically disconnected,
        regardless of
        the number of timesteps it is on overflow). This is called a "hard overflow". This is expressed in relative
        value of the thermal limits, for example, if for a powerline the `thermal_limit` is 150 and the
        HARD_OVERFLOW_THRESHOLD is 2.0, then if the flow on the powerline reaches 2 * 150 = 300.0 the powerline
        the powerline is automatically disconnected.

    SOFT_OVERFLOW_THRESHOLD: ``float``
        .. versionadded:: 1.9.3
        
        Threshold above which delayed protection are triggered. A line with its current bellow `SOFT_OVERFLOW_THRESHOLD * thermal_limit`
        then nothing happens. If it's above the delay start. And if it's above `SOFT_OVERFLOW_THRESHOLD * thermal_limit`
        for more than :attr:`NB_TIMESTEP_OVERFLOW_ALLOWED` consecutive steps.
    
    ENV_DC: ``bool``
        Whether or not making the simulations of the environment in the "direct current" approximation. This can be
        usefull for early training of agent, as this mode is much faster to compute than the corresponding
        "alternative current" powerflow. It is also less precise. The default is ``False``

    FORECAST_DC: ``bool``
        DEPRECATED. Please use the "change_forecast_param" function of the environment
        Whether to use the direct current approximation in the :func:`grid2op.Observation.BaseObservation.simulate`
        method. Default is ``False``. Setting :attr:`FORECAST_DC` to `True` can speed up the computation of the
        `simulate` function, but will make the results less accurate.

    MAX_SUB_CHANGED: ``int``
        Maximum number of substations that can be reconfigured between two consecutive timesteps by an
        :class:`grid2op.Agent.BaseAgent`. Default value is 1.

    MAX_LINE_STATUS_CHANGED: ``int``
        Maximum number of powerlines statuses that can be changed between two consecutive timesteps by an
        :class:`grid2op.Agent.BaseAgent`. Default value is 1.

    IGNORE_MIN_UP_DOWN_TIME: ``bool``
        Whether or not to ignore the attributes `gen_min_uptime` and `gen_min_downtime`. Basically setting this
        parameter to ``True``

    LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION: ``bool``
        If set to ``True`` (NOT the default) the environment will automatically limit the curtailment / storage actions that otherwise
        would lead to infeasible state.

        For example the maximum ramp up of generator at a given step is 100 MW / step (
        ie you can increase the production of these generators of maximum 100 MW at this step) but if you cumul the storage
        action and the curtailment action, you ask + 110 MW at this step (for example you curtail 100MW of renewables).

        In this example, if param.LIMIT_INFEASIBLE_CURTAILMENT_ACTION is ``False`` (default) this is a game over. If it's ``True``
        then the curtailment action is limited so that it does not exceed 100 MW.

        Setting it to ``True`` might help the learning of agent using redispatching.
        
        If you want a similar behaviour, when you don't have access to the parameters of the environment, you can
        have a look at :func:`grid2op.Aciton.BaseAction.limit_curtail_storage`. 
        
        .. note::
            This argument and the :func:`grid2op.Action.BaseAction.limit_curtail_storage` have the same objective:
            prevent an agent to do some curtailment too strong for the grid.
            
            When using this parameter, the environment will do it knowing exactly what will happen next (
                its a bit "cheating")  and limit exactly the action to exactly right amount.
            
            Using :func:`grid2op.Aciton.BaseAction.limit_curtail_storage` is always feasible, but less precise.

    INIT_STORAGE_CAPACITY: ``float``
        Between 0. and 1. Specify, at the beginning of each episode, what is the storage capacity of each storage unit.
        The storage capacity will be expressed as fraction of storage_Emax. For example, if `INIT_STORAGE_CAPACITY` is
        0.5 then at the beginning of every episode, all storage unit will have a storage capacity of
        0.5 * `storage_Emax`. By default: `0.5`

    ACTIVATE_STORAGE_LOSS: ``bool``
        You can set it to ``False`` to not take into account the loss in the storage units.
        This deactivates the "loss amount per time step" (`storage_loss`) and has also the effect to set
        to do **as if** the
        storage units were perfect (as if `storage_charging_efficiency=1.` and `storage_discharging_efficiency=1.`.

        **NB** it does **as if** it were the case. But the parameters `storage_loss`, `storage_charging_efficiency`
        and storage_discharging_efficiency` are not affected by this.

        Default: ``True``

    ALARM_BEST_TIME: ``int``
        Number of step for which it's best to send an alarm BEFORE a game over

    ALARM_WINDOW_SIZE: ``int``
        Number of steps for which it's worth it to give an alarm (if an alarm is send outside of the window
        `[ALARM_BEST_TIME - ALARM_WINDOW_SIZE, ALARM_BEST_TIME + ALARM_WINDOW_SIZE]` then it does not grant anything

    ALERT_TIME_WINDOW : ``int``
        Number of steps for which it's worth it to give an alert after an attack. If the alert is sent before, the assistant
        score doesn't take into account that an alert is raised. 

    MAX_SIMULATE_PER_STEP: ``int``
        Maximum number of calls to `obs.simuate(...)` allowed per step (reset each "env.step(...)"). Defaults to -1 meaning "as much as you want".

    MAX_SIMULATE_PER_EPISODE: ``int``
        Maximum number of calls to `obs.simuate(...)` allowed per episode (reset each "env.simulate(...)"). Defaults to -1 meaning "as much as you want".

    """

    def __init__(self, parameters_path=None):
        """
        Build an object representing the _parameters of the game.

        Parameters
        ----------
        parameters_path: ``str``, optional
            Path where to look for parameters.

        """
        # if True, then it will not disconnect lines above their thermal limits
        self.NO_OVERFLOW_DISCONNECTION = False

        # number of timestep before powerline with an overflow is automatically disconnected
        self.NB_TIMESTEP_OVERFLOW_ALLOWED = dt_int(2)

        # number of timestep before a line can be reconnected if it has suffer a forced disconnection
        self.NB_TIMESTEP_RECONNECTION = dt_int(10)

        # number of timestep before a substation topology can be modified again
        self.NB_TIMESTEP_COOLDOWN_LINE = dt_int(0)
        self.NB_TIMESTEP_COOLDOWN_SUB = dt_int(0)

        # threshold above which a powerline is instantly disconnected by protections
        # this is expressed in relative value of the thermal limits
        # for example setting "HARD_OVERFLOW_THRESHOLD = 2" is equivalent, if a powerline has a thermal limit of
        # 243 A, to disconnect it instantly if it has a powerflow higher than 2 * 243 = 486 A
        self.HARD_OVERFLOW_THRESHOLD = dt_float(2.0)
        
        self.SOFT_OVERFLOW_THRESHOLD = dt_float(1.0)

        # are the powerflow performed by the environment in DC mode (dc powerflow) or AC (ac powerflow)
        self.ENV_DC = False

        # same as above, but for the forecast states
        self.FORECAST_DC = False  # DEPRECATED use "change_forecast_parameters(new_param)" with "new_param.ENV_DC=..."

        # maximum number of substations that can be change in one action
        self.MAX_SUB_CHANGED = dt_int(1)

        # maximum number of powerline status that can be changed in one action
        self.MAX_LINE_STATUS_CHANGED = dt_int(1)

        # ignore the min_uptime and downtime for the generators: allow them to be connected / disconnected
        # at will
        self.IGNORE_MIN_UP_DOWN_TIME = True

        # allow dispatch on turned off generator (if ``True`` you can actually dispatch a turned on geenrator)
        self.ALLOW_DISPATCH_GEN_SWITCH_OFF = True

        # if a curtailment action is "too strong" it will limit it to the "maximum feasible"
        # not to break the whole system
        self.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = False

        # storage capacity (NOT in pct so 0.5 = 50%)
        self.INIT_STORAGE_CAPACITY = 0.5

        # do i take into account the storage loss in the step function
        self.ACTIVATE_STORAGE_LOSS = True

        # alarms
        self.ALARM_BEST_TIME = 12
        self.ALARM_WINDOW_SIZE = 12

        # alert 
        self.ALERT_TIME_WINDOW = 12

        # number of simulate
        self.MAX_SIMULATE_PER_STEP = dt_int(-1)
        self.MAX_SIMULATE_PER_EPISODE = dt_int(-1)

        if parameters_path is not None:
            if os.path.isfile(parameters_path):
                self.init_from_json(parameters_path)
            else:
                warn_msg = "Parameters: the file {} is not found. Continuing with default parameters."
                warnings.warn(warn_msg.format(parameters_path))

    @staticmethod
    def _isok_txt(arg):
        if isinstance(arg, type(True)):
            return arg
        if isinstance(arg, type("")):
            arg = arg.strip('"')
        elif isinstance(arg, type(1)):
            arg = "{}".format(arg)

        res = False
        if (
            arg == "True"
            or arg == "T"
            or arg == "true"
            or arg == "t"
            or str(arg) == "1"
        ):
            res = True
        elif (
            arg == "False"
            or arg == "F"
            or arg == "false"
            or arg == "f"
            or str(arg) == "0"
        ):
            res = False
        else:
            msg = (
                "It's ambiguous where an argument is True or False. "
                'Please only provide "True" or "False" and not {}'
            )
            raise RuntimeError(msg.format(arg))
        return res

    def init_from_dict(self, dict_):
        """
        Initialize the object given a dictionary. All keys are optional. If a key is not present in the dictionary,
        the default parameters is used.

        Parameters
        ----------
        dict_: ``dict``
            The dictionary representing the parameters to load.

        """
        if "NO_OVERFLOW_DISCONNECTION" in dict_:
            self.NO_OVERFLOW_DISCONNECTION = Parameters._isok_txt(
                dict_["NO_OVERFLOW_DISCONNECTION"]
            )

        if "IGNORE_MIN_UP_DOWN_TIME" in dict_:
            self.IGNORE_MIN_UP_DOWN_TIME = Parameters._isok_txt(
                dict_["IGNORE_MIN_UP_DOWN_TIME"]
            )
        if "ALLOW_DISPATCH_GEN_SWITCH_OFF" in dict_:
            self.ALLOW_DISPATCH_GEN_SWITCH_OFF = Parameters._isok_txt(
                dict_["ALLOW_DISPATCH_GEN_SWITCH_OFF"]
            )
        if "LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION" in dict_:
            self.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = Parameters._isok_txt(
                dict_["LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION"]
            )

        if "NB_TIMESTEP_POWERFLOW_ALLOWED" in dict_:
            self.NB_TIMESTEP_OVERFLOW_ALLOWED = dt_int(
                dict_["NB_TIMESTEP_POWERFLOW_ALLOWED"]
            )
        if "NB_TIMESTEP_OVERFLOW_ALLOWED" in dict_:
            self.NB_TIMESTEP_OVERFLOW_ALLOWED = dt_int(
                dict_["NB_TIMESTEP_OVERFLOW_ALLOWED"]
            )

        if "NB_TIMESTEP_RECONNECTION" in dict_:
            self.NB_TIMESTEP_RECONNECTION = dt_int(dict_["NB_TIMESTEP_RECONNECTION"])

        if "HARD_OVERFLOW_THRESHOLD" in dict_:
            self.HARD_OVERFLOW_THRESHOLD = dt_float(dict_["HARD_OVERFLOW_THRESHOLD"])
            
        if "SOFT_OVERFLOW_THRESHOLD" in dict_:
            self.SOFT_OVERFLOW_THRESHOLD = dt_float(dict_["SOFT_OVERFLOW_THRESHOLD"])

        if "ENV_DC" in dict_:
            self.ENV_DC = Parameters._isok_txt(dict_["ENV_DC"])

        if "FORECAST_DC" in dict_:
            new_val = Parameters._isok_txt(dict_["FORECAST_DC"])
            if new_val != self.FORECAST_DC:
                warnings.warn(
                    "The FORECAST_DC attributes is deprecated. Please change the parameters of the "
                    '"forecast" backend with "env.change_forecast_parameters(new_param)" function '
                    'with "new_param.ENV_DC=..." '
                )
            self.FORECAST_DC = new_val

        if "MAX_SUB_CHANGED" in dict_:
            self.MAX_SUB_CHANGED = dt_int(dict_["MAX_SUB_CHANGED"])

        if "MAX_LINE_STATUS_CHANGED" in dict_:
            self.MAX_LINE_STATUS_CHANGED = dt_int(dict_["MAX_LINE_STATUS_CHANGED"])

        if "NB_TIMESTEP_TOPOLOGY_REMODIF" in dict_:
            # for backward compatibility (in case of old dataset)
            self.NB_TIMESTEP_COOLDOWN_SUB = dt_int(
                dict_["NB_TIMESTEP_TOPOLOGY_REMODIF"]
            )
        if "NB_TIMESTEP_COOLDOWN_SUB" in dict_:
            self.NB_TIMESTEP_COOLDOWN_SUB = dt_int(dict_["NB_TIMESTEP_COOLDOWN_SUB"])

        if "NB_TIMESTEP_LINE_STATUS_REMODIF" in dict_:
            # for backward compatibility (in case of old dataset)
            self.NB_TIMESTEP_COOLDOWN_LINE = dt_int(
                dict_["NB_TIMESTEP_LINE_STATUS_REMODIF"]
            )
        if "NB_TIMESTEP_COOLDOWN_LINE" in dict_:
            self.NB_TIMESTEP_COOLDOWN_LINE = dt_int(dict_["NB_TIMESTEP_COOLDOWN_LINE"])

        # storage parameters
        if "INIT_STORAGE_CAPACITY" in dict_:
            self.INIT_STORAGE_CAPACITY = dt_float(dict_["INIT_STORAGE_CAPACITY"])
        if "ACTIVATE_STORAGE_LOSS" in dict_:
            self.ACTIVATE_STORAGE_LOSS = Parameters._isok_txt(
                dict_["ACTIVATE_STORAGE_LOSS"]
            )

        # alarm parameters
        if "ALARM_BEST_TIME" in dict_:
            self.ALARM_BEST_TIME = dt_int(dict_["ALARM_BEST_TIME"])
        if "ALARM_WINDOW_SIZE" in dict_:
            self.ALARM_WINDOW_SIZE = dt_int(dict_["ALARM_WINDOW_SIZE"])

        # alert parameters 
        if "ALERT_TIME_WINDOW" in dict_:
            self.ALERT_TIME_WINDOW = dt_int(dict_["ALERT_TIME_WINDOW"])

        if "MAX_SIMULATE_PER_STEP" in dict_:
            self.MAX_SIMULATE_PER_STEP = dt_int(dict_["MAX_SIMULATE_PER_STEP"])

        if "MAX_SIMULATE_PER_EPISODE" in dict_:
            self.MAX_SIMULATE_PER_EPISODE = dt_int(dict_["MAX_SIMULATE_PER_EPISODE"])

        authorized_keys = set(self.__dict__.keys())
        authorized_keys = authorized_keys | {
            "NB_TIMESTEP_POWERFLOW_ALLOWED",
            "NB_TIMESTEP_TOPOLOGY_REMODIF",
            "NB_TIMESTEP_LINE_STATUS_REMODIF",
        }

        ignored_keys = dict_.keys() - authorized_keys
        if len(ignored_keys):
            warnings.warn(
                'Parameters: The _parameters "{}" used to build the Grid2Op.Parameters '
                "class are not recognized and will be ignored.".format(ignored_keys)
            )

    def to_dict(self):
        """
        Serialize all the _parameters as a dictionnary; Useful to write it in json format.

        Returns
        -------
        res: ``dict``
            A representation of these _parameters in the form of a dictionnary.

        """
        res = {}
        res["NO_OVERFLOW_DISCONNECTION"] = bool(self.NO_OVERFLOW_DISCONNECTION)
        res["IGNORE_MIN_UP_DOWN_TIME"] = bool(self.IGNORE_MIN_UP_DOWN_TIME)
        res["ALLOW_DISPATCH_GEN_SWITCH_OFF"] = bool(self.ALLOW_DISPATCH_GEN_SWITCH_OFF)
        res["LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION"] = bool(
            self.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION
        )
        res["NB_TIMESTEP_OVERFLOW_ALLOWED"] = int(self.NB_TIMESTEP_OVERFLOW_ALLOWED)
        res["NB_TIMESTEP_RECONNECTION"] = int(self.NB_TIMESTEP_RECONNECTION)
        res["HARD_OVERFLOW_THRESHOLD"] = float(self.HARD_OVERFLOW_THRESHOLD)
        res["SOFT_OVERFLOW_THRESHOLD"] = float(self.SOFT_OVERFLOW_THRESHOLD)
        res["ENV_DC"] = bool(self.ENV_DC)
        res["FORECAST_DC"] = bool(self.FORECAST_DC)
        res["MAX_SUB_CHANGED"] = int(self.MAX_SUB_CHANGED)
        res["MAX_LINE_STATUS_CHANGED"] = int(self.MAX_LINE_STATUS_CHANGED)
        res["NB_TIMESTEP_COOLDOWN_LINE"] = int(self.NB_TIMESTEP_COOLDOWN_LINE)
        res["NB_TIMESTEP_COOLDOWN_SUB"] = int(self.NB_TIMESTEP_COOLDOWN_SUB)
        res["INIT_STORAGE_CAPACITY"] = float(self.INIT_STORAGE_CAPACITY)
        res["ACTIVATE_STORAGE_LOSS"] = bool(self.ACTIVATE_STORAGE_LOSS)
        res["ALARM_BEST_TIME"] = int(self.ALARM_BEST_TIME)
        res["ALARM_WINDOW_SIZE"] = int(self.ALARM_WINDOW_SIZE)
        res["ALERT_TIME_WINDOW"] = int(self.ALERT_TIME_WINDOW)
        res["MAX_SIMULATE_PER_STEP"] = int(self.MAX_SIMULATE_PER_STEP)
        res["MAX_SIMULATE_PER_EPISODE"] = int(self.MAX_SIMULATE_PER_EPISODE)
        return res

    def init_from_json(self, json_path):
        """
        Set member attributes from a json file

        Parameters
        ----------
        json_path: ``str``
            The complete (*ie.* path + filename) where the json file is located.
        """
        try:
            with open(json_path) as f:
                dict_ = json.load(f)
            self.init_from_dict(dict_)
        except Exception as exc_:
            warn_msg = (
                "Could not load from {}\n"
                'Continuing with default parameters. \n\nThe error was "{}"'
            )
            warnings.warn(warn_msg.format(json_path, exc_))

    def __eq__(self, other):
        this_dict = self.to_dict()
        other_dict = other.to_dict()
        return this_dict == other_dict

    @staticmethod
    def from_json(json_path):
        """
        Create instance of a Parameters from a path where is a json is saved.

        Parameters
        ----------
        json_path: ``str``
            The complete (*ie.* path + filename) where the json file is located.

        Returns
        -------
        res: :class:`Parameters`
            The _parameters initialized

        """
        res = Parameters(json_path)
        return res

    def check_valid(self):
        """

        check the parameter is valid (ie it checks that all the values are of correct types and within the
        correct range.

        Raises
        -------
        An exception if the parameter is not valid
        """
        try:
            if not isinstance(self.NO_OVERFLOW_DISCONNECTION, (bool, dt_bool)):
                raise RuntimeError("NO_OVERFLOW_DISCONNECTION should be a boolean")
            self.NO_OVERFLOW_DISCONNECTION = dt_bool(self.NO_OVERFLOW_DISCONNECTION)
        except Exception as exc_:
            raise RuntimeError(
                f'Impossible to convert NO_OVERFLOW_DISCONNECTION to bool with error \n:"{exc_}"'
            )

        try:
            self.NB_TIMESTEP_OVERFLOW_ALLOWED = int(
                self.NB_TIMESTEP_OVERFLOW_ALLOWED
            )  # to raise if numpy array
            self.NB_TIMESTEP_OVERFLOW_ALLOWED = dt_int(
                self.NB_TIMESTEP_OVERFLOW_ALLOWED
            )
        except Exception as exc_:
            raise RuntimeError(
                f'Impossible to convert NB_TIMESTEP_OVERFLOW_ALLOWED to int with error \n:"{exc_}"'
            )

        if self.NB_TIMESTEP_OVERFLOW_ALLOWED < 0:
            raise RuntimeError(
                "NB_TIMESTEP_OVERFLOW_ALLOWED < 0., this should be >= 0."
            )
        try:
            self.NB_TIMESTEP_RECONNECTION = int(
                self.NB_TIMESTEP_RECONNECTION
            )  # to raise if numpy array
            self.NB_TIMESTEP_RECONNECTION = dt_int(self.NB_TIMESTEP_RECONNECTION)
        except Exception as exc_:
            raise RuntimeError(
                f'Impossible to convert NB_TIMESTEP_RECONNECTION to int with error \n:"{exc_}"'
            )
        if self.NB_TIMESTEP_RECONNECTION < 0:
            raise RuntimeError("NB_TIMESTEP_RECONNECTION < 0., this should be >= 0.")
        try:
            self.NB_TIMESTEP_COOLDOWN_LINE = int(self.NB_TIMESTEP_COOLDOWN_LINE)
            self.NB_TIMESTEP_COOLDOWN_LINE = dt_int(self.NB_TIMESTEP_COOLDOWN_LINE)
        except Exception as exc_:
            raise RuntimeError(
                f'Impossible to convert NB_TIMESTEP_COOLDOWN_LINE to int with error \n:"{exc_}"'
            )
        if self.NB_TIMESTEP_COOLDOWN_LINE < 0:
            raise RuntimeError("NB_TIMESTEP_COOLDOWN_LINE < 0., this should be >= 0.")
        try:
            self.NB_TIMESTEP_COOLDOWN_SUB = int(
                self.NB_TIMESTEP_COOLDOWN_SUB
            )  # to raise if numpy array
            self.NB_TIMESTEP_COOLDOWN_SUB = dt_int(self.NB_TIMESTEP_COOLDOWN_SUB)
        except Exception as exc_:
            raise RuntimeError(
                f'Impossible to convert NB_TIMESTEP_COOLDOWN_SUB to int with error \n:"{exc_}"'
            )
        if self.NB_TIMESTEP_COOLDOWN_SUB < 0:
            raise RuntimeError("NB_TIMESTEP_COOLDOWN_SUB < 0., this should be >= 0.")
        try:
            self.HARD_OVERFLOW_THRESHOLD = float(
                self.HARD_OVERFLOW_THRESHOLD
            )  # to raise if numpy array
            self.HARD_OVERFLOW_THRESHOLD = dt_float(self.HARD_OVERFLOW_THRESHOLD)
        except Exception as exc_:
            raise RuntimeError(
                f'Impossible to convert HARD_OVERFLOW_THRESHOLD to float with error \n:"{exc_}"'
            )
        if self.HARD_OVERFLOW_THRESHOLD < 1.0:
            raise RuntimeError(
                "HARD_OVERFLOW_THRESHOLD < 1., this should be >= 1. (use env.set_thermal_limit "
                "to modify the thermal limit)"
            )
            
        try:
            self.SOFT_OVERFLOW_THRESHOLD = float(
                self.SOFT_OVERFLOW_THRESHOLD
            )  # to raise if numpy array
            self.SOFT_OVERFLOW_THRESHOLD = dt_float(self.SOFT_OVERFLOW_THRESHOLD)
        except Exception as exc_:
            raise RuntimeError(
                f'Impossible to convert SOFT_OVERFLOW_THRESHOLD to float with error \n:"{exc_}"'
            )
        if self.SOFT_OVERFLOW_THRESHOLD < 1.0:
            raise RuntimeError(
                "SOFT_OVERFLOW_THRESHOLD < 1., this should be >= 1. (use env.set_thermal_limit "
                "to modify the thermal limit)"
            )
        if self.SOFT_OVERFLOW_THRESHOLD >= self.HARD_OVERFLOW_THRESHOLD:
            raise RuntimeError(
                "self.SOFT_OVERFLOW_THRESHOLD >= self.HARD_OVERFLOW_THRESHOLD this would that the"
                "soft overflow would be deactivated. It's not possible at the moment."
            )
            
        try:
            if not isinstance(self.ENV_DC, (bool, dt_bool)):
                raise RuntimeError("NO_OVERFLOW_DISCONNECTION should be a boolean")
            self.ENV_DC = dt_bool(self.ENV_DC)
        except Exception as exc_:
            raise RuntimeError(
                f'Impossible to convert ENV_DC to bool with error \n:"{exc_}"'
            )
        try:
            self.MAX_SUB_CHANGED = int(self.MAX_SUB_CHANGED)  # to raise if numpy array
            self.MAX_SUB_CHANGED = dt_int(self.MAX_SUB_CHANGED)
        except Exception as exc_:
            raise RuntimeError(
                f'Impossible to convert MAX_SUB_CHANGED to int with error \n:"{exc_}"'
            )
        if self.MAX_SUB_CHANGED < 0:
            raise RuntimeError(
                "MAX_SUB_CHANGED should be >=0 (or -1 if you want to be able to change every "
                "substation at once)"
            )
        try:
            self.MAX_LINE_STATUS_CHANGED = int(
                self.MAX_LINE_STATUS_CHANGED
            )  # to raise if numpy array
            self.MAX_LINE_STATUS_CHANGED = dt_int(self.MAX_LINE_STATUS_CHANGED)
        except Exception as exc_:
            raise RuntimeError(
                f'Impossible to convert MAX_LINE_STATUS_CHANGED to int with error \n:"{exc_}"'
            )
        if self.MAX_LINE_STATUS_CHANGED < 0:
            raise RuntimeError(
                "MAX_LINE_STATUS_CHANGED should be >=0 "
                "(or -1 if you want to be able to change every powerline at once)"
            )
        try:
            if not isinstance(self.IGNORE_MIN_UP_DOWN_TIME, (bool, dt_bool)):
                raise RuntimeError("IGNORE_MIN_UP_DOWN_TIME should be a boolean")
            self.IGNORE_MIN_UP_DOWN_TIME = dt_bool(self.IGNORE_MIN_UP_DOWN_TIME)
        except Exception as exc_:
            raise RuntimeError(
                f'Impossible to convert IGNORE_MIN_UP_DOWN_TIME to bool with error \n:"{exc_}"'
            )
        try:
            if not isinstance(self.ALLOW_DISPATCH_GEN_SWITCH_OFF, (bool, dt_bool)):
                raise RuntimeError("ALLOW_DISPATCH_GEN_SWITCH_OFF should be a boolean")
            self.ALLOW_DISPATCH_GEN_SWITCH_OFF = dt_bool(
                self.ALLOW_DISPATCH_GEN_SWITCH_OFF
            )
        except Exception as exc_:
            raise RuntimeError(
                f'Impossible to convert ALLOW_DISPATCH_GEN_SWITCH_OFF to bool with error \n:"{exc_}"'
            )
        try:
            if not isinstance(
                self.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION, (bool, dt_bool)
            ):
                raise RuntimeError(
                    "LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION should be a boolean"
                )
            self.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = dt_bool(
                self.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION
            )
        except Exception as exc_:
            raise RuntimeError(
                f'Impossible to convert LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION to bool with error \n:"{exc_}"'
            )

        try:
            self.INIT_STORAGE_CAPACITY = float(
                self.INIT_STORAGE_CAPACITY
            )  # to raise if numpy array
            self.INIT_STORAGE_CAPACITY = dt_float(self.INIT_STORAGE_CAPACITY)
        except Exception as exc_:
            raise RuntimeError(
                f'Impossible to convert INIT_STORAGE_CAPACITY to float with error \n:"{exc_}"'
            )

        if self.INIT_STORAGE_CAPACITY < 0.0:
            raise RuntimeError(
                "INIT_STORAGE_CAPACITY < 0., this should be within range [0., 1.]"
            )
        if self.INIT_STORAGE_CAPACITY > 1.0:
            raise RuntimeError(
                "INIT_STORAGE_CAPACITY > 1., this should be within range [0., 1.]"
            )

        try:
            if not isinstance(self.ACTIVATE_STORAGE_LOSS, (bool, dt_bool)):
                raise RuntimeError("ACTIVATE_STORAGE_LOSS should be a boolean")
            self.ACTIVATE_STORAGE_LOSS = dt_bool(self.ACTIVATE_STORAGE_LOSS)
        except Exception as exc_:
            raise RuntimeError(
                f'Impossible to convert ACTIVATE_STORAGE_LOSS to bool with error \n:"{exc_}"'
            )

        try:
            self.ALARM_WINDOW_SIZE = dt_int(self.ALARM_WINDOW_SIZE)
        except Exception as exc_:
            raise RuntimeError(
                f'Impossible to convert ALARM_WINDOW_SIZE to int with error \n:"{exc_}"'
            )
        try:
            self.ALARM_BEST_TIME = dt_int(self.ALARM_BEST_TIME)
        except Exception as exc_:
            raise RuntimeError(
                f'Impossible to convert ALARM_BEST_TIME to int with error \n:"{exc_}"'
            )
        try:
            self.ALERT_TIME_WINDOW = dt_int(self.ALERT_TIME_WINDOW)
        except Exception as exc_:
            raise RuntimeError(
                f'Impossible to convert ALERT_TIME_WINDOW to int with error \n:"{exc_}"'
            )

        if self.ALARM_WINDOW_SIZE <= 0:
            raise RuntimeError("self.ALARM_WINDOW_SIZE should be a positive integer !")
        if self.ALARM_BEST_TIME <= 0:
            raise RuntimeError("self.ALARM_BEST_TIME should be a positive integer !")
        if self.ALERT_TIME_WINDOW <= 0:
            raise RuntimeError("self.ALERT_TIME_WINDOW should be a positive integer !")

        try:
            self.MAX_SIMULATE_PER_STEP = int(
                self.MAX_SIMULATE_PER_STEP
            )  # to raise if numpy array
            self.MAX_SIMULATE_PER_STEP = dt_int(self.MAX_SIMULATE_PER_STEP)
        except Exception as exc_:
            raise RuntimeError(
                f'Impossible to convert MAX_SIMULATE_PER_STEP to int with error \n:"{exc_}"'
            )
        if self.MAX_SIMULATE_PER_STEP <= -2:
            raise RuntimeError(
                f"self.MAX_SIMULATE_PER_STEP should be a positive integer or -1, we found {self.MAX_SIMULATE_PER_STEP}"
            )

        try:
            self.MAX_SIMULATE_PER_EPISODE = int(
                self.MAX_SIMULATE_PER_EPISODE
            )  # to raise if numpy array
            self.MAX_SIMULATE_PER_EPISODE = dt_int(self.MAX_SIMULATE_PER_EPISODE)
        except Exception as exc_:
            raise RuntimeError(
                f'Impossible to convert MAX_SIMULATE_PER_EPISODE to int with error \n:"{exc_}"'
            )
        if self.MAX_SIMULATE_PER_EPISODE <= -2:
            raise RuntimeError(
                f"self.MAX_SIMULATE_PER_EPISODE should be a positive integer or -1, we found {self.MAX_SIMULATE_PER_EPISODE}"
            )
