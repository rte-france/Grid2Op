# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from collections.abc import Iterable
import numpy as np

from grid2op.MakeEnv import make
try:
    from lightsim2grid import LightSimBackend
    bkclass = LightSimBackend
    # raise ImportError()
except ImportError as excq_:
    from grid2op.Backend import PandaPowerBackend
    bkclass = PandaPowerBackend
    pass


class EnvCache(object):
    """
    TODO this is not implemented yet (at least the "cache" part)

    We should have a flag that when an environment is computing, it returns an "error" to indicate that on the person
    who makes the call (for now i'm pretty sure the current implementation will not work in asynch mode).
    """
    ENV_NOT_FOUND = 0
    ENV_ID_NOT_FOUND = 1
    INVALID_ACTION = 2
    INVALID_STEP = 3
    ERROR_ENV_SEED = 4
    ERROR_ENV_RESET = 5
    ERROR_ENV_SET_ID = 6
    ERROR_ENV_THERMAL_LIMIT = 7
    ERROR_CLOSE = 8
    ERROR_ENV_FAST_FORWARD = 9
    ERROR_ENV_PATH = 10

    def __init__(self, ujson_as_json):
        self.all_env = {}
        self.ujson_as_json = ujson_as_json  # do i use the faster "ujson" library to parse json
        self._convert_json = not self.ujson_as_json

    def insert_env(self, env_name):
        """
        TODO
        """
        try:
            env = make(env_name, backend=bkclass())  # TODO look at the RemoteEnv here
            env.deactivate_forecast()
            # call deactivate_forecast

        except Exception as exc_:
            return None, None, exc_

        if env_name not in self.all_env:
            # create an environment with that name
            self.all_env[env_name] = [env]
        else:
            self.all_env[env_name].append(env)
        id_ = len(self.all_env[env_name]) - 1
        obs = env.reset()
        return id_, obs.to_json(convert=self._convert_json), None

    def step(self, env_name, env_id, action_as_json):
        """
        TODO
        """
        res_env = (None, None, None, None)
        env, (error_id, error_msg) = self._aux_get_env(env_name, env_id)

        if error_id is not None:
            return res_env, (error_id, error_msg)
        try:
            act = env.action_space()
            act.from_json(action_as_json)
        except Exception as exc_:
            msg_ = f"impossible to convert the provided action to a valid action on this environment with error:\n" \
                   f"{exc_}"
            return res_env, (self.INVALID_ACTION, msg_)

        try:
            obs, reward, done, info = env.step(act)
        except Exception as exc_:
            msg_ = f"impossible to make a step on the give environment with error\n{exc_}"
            return res_env, (self.INVALID_STEP, msg_)
        return (obs.to_json(convert=self._convert_json),
                float(reward),
                bool(done),
                self._aux_info_to_json(info)), (None, None)

    def seed(self, env_name, env_id, seed):
        """
        TODO
        """
        env, (error_id, error_msg) = self._aux_get_env(env_name, env_id)

        if error_id is not None:
            return None, (error_id, error_msg)
        try:
            seeds = env.seed(seed)
        except Exception as exc_:
            msg_ = f"Impossible to seed the environment with error:\n{exc_}"
            return None, (self.ERROR_ENV_SEED, msg_)
        return self._aux_array_to_json(seeds), (None, None)

    def reset(self, env_name, env_id):
        """
        TODO
        """
        env, (error_id, error_msg) = self._aux_get_env(env_name, env_id)
        if error_id is not None:
            return None, (error_id, error_msg)

        try:
            obs = env.reset()
        except Exception as exc_:
            msg_ = f"Impossible to reset the environment with error {exc_}"
            return None, (self.ERROR_ENV_RESET, msg_)

        return obs.to_json(convert=self._convert_json), (None, None)

    def set_id(self, env_name, env_id, chron_id):
        """
        TODO
        """
        env, (error_id, error_msg) = self._aux_get_env(env_name, env_id)
        if error_id is not None:
            return error_id, error_msg

        try:
            env.set_id(chron_id)
        except Exception as exc_:
            msg_ = f"Impossible to set the chronics id of the environment with error:\n {exc_}"
            return self.ERROR_ENV_SET_ID, msg_
        return None, None

    def set_thermal_limit(self, env_name, env_id, thermal_limit):
        """
        TODO
        """
        env, (error_id, error_msg) = self._aux_get_env(env_name, env_id)
        if error_id is not None:
            return error_id, error_msg

        try:
            env.set_thermal_limit(thermal_limit)
        except Exception as exc_:
            msg_ = f"Impossible to set the thermal limits of the environment with error:\n {exc_}"
            return self.ERROR_ENV_THERMAL_LIMIT, msg_
        return None, None

    def get_thermal_limit(self, env_name, env_id):
        """
        TODO
        """
        env, (error_id, error_msg) = self._aux_get_env(env_name, env_id)
        if error_id is not None:
            return None, (error_id, error_msg)

        try:
            res = env.get_thermal_limit()
        except Exception as exc_:
            msg_ = f"Impossible to get the thermal limits of the environment with error:\n {exc_}"
            return None, (self.ERROR_ENV_THERMAL_LIMIT, msg_)
        res = res.tolist()
        return res, (None, None)

    def close(self, env_name, env_id):
        """
        TODO
        """
        env, (error_id, error_msg) = self._aux_get_env(env_name, env_id)
        if error_id is not None:
            return error_id, error_msg

        try:
            env.close()
        except Exception as exc_:
            msg_ = f"Impossible to close the environment with error:\n {exc_}"
            return self.ERROR_CLOSE, msg_
        return None, None

    def fast_forward_chronics(self, env_name, env_id, nb_step):
        """
        TODO
        """
        env, (error_id, error_msg) = self._aux_get_env(env_name, env_id)
        if error_id is not None:
            return error_id, error_msg

        try:
            env.fast_forward_chronics(nb_step)
        except Exception as exc_:
            msg_ = f"Impossible to fast forward the environment with error:\n {exc_}"
            return self.ERROR_ENV_FAST_FORWARD, msg_
        return None, None

    def get_path_env(self, env_name, env_id):
        """
        TODO
        """
        env, (error_id, error_msg) = self._aux_get_env(env_name, env_id)
        if error_id is not None:
            return None, (error_id, error_msg)

        try:
            res = env.get_path_env()
        except Exception as exc_:
            msg_ = f"Impossible to fast forward the environment with error:\n {exc_}"
            return None, (self.ERROR_ENV_PATH, msg_)
        return res, (None, None)

    def train_val_split(self, env_name, env_id, id_chron_val):
        """
        TODO
        """
        env, (error_id, error_msg) = self._aux_get_env(env_name, env_id)
        if error_id is not None:
            return (None, None), (error_id, error_msg)

        try:
            res = env.train_val_split(val_scen_id=id_chron_val,
                                      add_for_train="train", add_for_val="val"
                                      )
        except Exception as exc_:
            msg_ = f"Impossible to split the environment with error:\n {exc_}"
            return (None, None), (self.ERROR_ENV_PATH, msg_)
        return res, (None, None)

    def _aux_array_to_json(self, array):
        if isinstance(array, Iterable):
            res = None
            if isinstance(array, np.ndarray):
                if array.shape == ():
                    res = []
            if res is None:
                res = [self._aux_array_to_json(el) for el in array]
            return res
        else:
            return float(array)

    def _aux_info_to_json(self, info):
        # TODO
        res = {}
        res["disc_lines"] = [int(el) for el in info["disc_lines"]]
        res["is_illegal"] = bool(info["is_illegal"])
        res["is_ambiguous"] = bool(info["is_ambiguous"])
        res["is_dispatching_illegal"] = bool(info["is_dispatching_illegal"])
        res["is_illegal_reco"] = bool(info["is_illegal_reco"])
        if info["opponent_attack_line"] is not None:
            res["opponent_attack_line"] = [bool(el) for el in info["opponent_attack_line"]]
        else:
            res["opponent_attack_line"] = None
        res["exception"] = [f"{exc_}" for exc_ in info["exception"]]
        return res

    def _aux_get_env(self, env_name, env_id):
        if env_name not in self.all_env:
            return None, (self.ENV_NOT_FOUND, f"environment \"{env_name}\" does not exists")

        li_env = self.all_env[env_name]
        env_id = int(env_id)
        nb_env = len(li_env)
        if env_id >= nb_env:
            msg_ = f"you asked to run the environment {env_id}  of {env_name}. But there are only {nb_env} " \
                   f"such environments"
            return None, (self.ENV_ID_NOT_FOUND, msg_)
        env = li_env[env_id]
        return env, (None, None)
