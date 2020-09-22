# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import copy
import json
import os
import re
import numpy as np

from datetime import timedelta

from grid2op.dtypes import dt_float, dt_int, dt_bool
from grid2op.Exceptions import Grid2OpException
from grid2op.Chronics import GridValue, ChronicsHandler
from grid2op.Opponent import BaseOpponent
from grid2op.Environment import Environment

from grid2op.Episode.EpisodeData import EpisodeData


class _GridFromLog(GridValue):
    """
    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    """
    def __init__(self, episode_data,
                 time_interval=timedelta(minutes=5),
                 max_iter=-1,
                 start_datetime=None,
                 chunk_size=None
                 ):
        # TODO reload directly the loadp, loadq, prodp and prodv from the path of the episode data if possible
        self.episode_data = episode_data
        if start_datetime is None:
            warnings.warn("\"start_datetime\" argument is ignored when building the _GridFromLog")
        if chunk_size is None:
            warnings.warn("\"chunk_size\" argument is ignored when building the _GridFromLog")
        GridValue.__init__(self,
                           time_interval=time_interval,
                           max_iter=max_iter,
                           start_datetime=self.episode_data.observations[0].get_time_stamp(),
                           chunk_size=None)

        # TODO reload that
        self.maintenance_time = np.zeros(self.episode_data.observations[0].line_status.shape[0], dtype=int) - 1
        self.maintenance_duration = np.zeros(self.episode_data.observations[0].line_status.shape[0], dtype=int)
        self.hazard_duration = np.zeros(self.episode_data.observations[0].line_status.shape[0], dtype=int)
        self.curr_iter = 0

    def initialize(self, order_backend_loads, order_backend_prods, order_backend_lines, order_backend_subs,
                   names_chronics_to_backend):
        pass

    def load_next(self):
        self.curr_iter += 1
        obs = self.episode_data.observations[self.curr_iter]
        self.current_datetime = obs.get_time_stamp()

        res = {}
        injs = {"prod_p": obs.prod_p.astype(dt_float),
                "load_p": obs.load_p.astype(dt_float),
                "load_q": obs.load_q.astype(dt_float),
                }
        res["injection"] = injs

        # TODO
        # if self.maintenance is not None:
        #     res["maintenance"] = self.maintenance[self.current_index, :]
        # if self.hazards is not None:
        #     res["hazards"] = self.hazards[self.current_index, :]

        prod_v = obs.prod_v
        return self.current_datetime,\
               res, \
               self.maintenance_time, \
               self.maintenance_duration, \
               self.hazard_duration, \
               prod_v

    def check_validity(self, backend):
        return True

    def next_chronics(self):
        self.episode_data.reboot()


class OpponentFromLog(BaseOpponent):
    """
    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    """
    pass


class EpisodeReboot:
    """
    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
        This is a first implementation to serve as "what can be done".

        It is a beta feature

    """
    def __init__(self):
        self.episode_data = None
        self.env = None
        self.chronics_handler = None
        self.current_time_step = None
        self.action = None  # the last action played

        warnings.warn("EpisodeReboot is a beta feature, it will likely be renamed, methods will be adapted "
                      "and it has probably some bugs. Use with care!")

    def load(self, backend, agent_path=None, name=None, data=None, env_kwargs={}):
        if data is None:
            if agent_path is not None and name is not None:
                self.episode_data = EpisodeData.from_disk(agent_path, name)
            else:
                raise Grid2OpException("To replay an episode you need at least to provide an EpisodeData "
                                       "(using the keyword argument \"data=...\") or provide the path and name where "
                                       "the "
                                       "episode is stored (keyword arguments \"agent_path\" and \"name\").")
        else:
            self.episode_data = copy.deepcopy(data)
            self.episode_data.reboot()

        self.chronics_handler = ChronicsHandler(chronicsClass=_GridFromLog,
                                                episode_data=self.episode_data)

        if "chronics_handler" in env_kwargs:
            del env_kwargs["chronics_handler"]
        if "backend" in env_kwargs:
            del env_kwargs["backend"]
        if "opponent_class" in env_kwargs:
            del env_kwargs["opponent_class"]
        if "name" in env_kwargs:
            del env_kwargs["name"]

        nm = "unknonwn"
        seed = None
        with open(os.path.join(agent_path, name, "episode_meta.json")) as f:
            dict_ = json.load(f)
            nm = re.sub("Environment_", "", dict_["env_type"])
            if dict_["env_seed"] is not None:
                seed = int(dict_["env_seed"])

        self.env = Environment(**env_kwargs,
                               backend=backend,
                               chronics_handler=self.chronics_handler,
                               opponent_class=OpponentFromLog,
                               name=nm)
        if seed is not None:
            self.env.seed(seed)

        tmp = self.env.reset()

        # always have the two bellow synch ! otherwise it messes up the "chronics"
        # in the env, when calling "env.step"
        self.current_time_step = 0
        self.env.chronics_handler.real_data.curr_iter = 0

        # first observation of the scenario
        current_obs = self.episode_data.observations[self.current_time_step]
        self._assign_state(current_obs)
        return self.env.get_obs()

    def _assign_state(self, obs):
        """
        works only if observation store the complete state of the grid...
        """
        self.env._gen_activeprod_t[:] = obs.prod_p.astype(dt_float)
        self.env._actual_dispatch[:] = obs.actual_dispatch.astype(dt_float)
        self.env._target_dispatch[:] = obs.target_dispatch.astype(dt_float)
        self.env._gen_activeprod_t_redisp[:] = obs.prod_p.astype(dt_float) + obs.actual_dispatch.astype(dt_float)
        self.env.current_obs = obs
        self.env._timestep_overflow[:] = obs.timestep_overflow.astype(dt_int)
        self.env._times_before_line_status_actionable[:] = obs.time_before_cooldown_line.astype(dt_int)
        self.env._times_before_topology_actionable[:] = obs.time_before_cooldown_sub.astype(dt_int)

        self.env._duration_next_maintenance[:] = obs.duration_next_maintenance.astype(dt_int)
        self.env._time_next_maintenance[:] = obs.time_next_maintenance.astype(dt_int)

        # TODO check that the "stored" "last bus for when the powerline were connected" are
        # kept there (I might need to do a for loop)
        # to test that i might need to use a "change status" and see if powerlines are connected
        # to the right bus
        self.env._backend_action += self.env._helper_action_env({"set_bus": obs.topo_vect.astype(dt_int),
                                                                 "injection": {"load_p": obs.load_p.astype(dt_float),
                                                                               "load_q": obs.load_q.astype(dt_float),
                                                                               "prod_p": obs.prod_p.astype(dt_float),
                                                                               "prod_v": obs.prod_v.astype(dt_float),
                                                                               }
                                                                 })
        self.env.backend.apply_action(self.env._backend_action)
        disc_lines, detailed_info, conv_ = self.env.backend.next_grid_state(env=self.env)
        if conv_ is None:
            self.env._backend_action.update_state(disc_lines)
        self.env._backend_action.reset()

    def next(self, update=False):
        """
        go to next time step
        if "update" then i reuse the observation stored to go to this time step, otherwise not

        do as if the environment will execute the action the stored agent did at the next time step
        (compared to the time step the environment is currently at)
        """
        if self.current_time_step is None:
            raise Grid2OpException("Impossible to go to the next time step with an episode not loaded. "
                                   "Call \"EpisodeReboot.load\" before.")

        if update:
            # I put myself at the observation just before the next time step
            obs = self.episode_data.observations[self.current_time_step]
            self.env._backend_action = self.env._backend_action_class()
            self._assign_state(obs)

        self.action = self.episode_data.actions[self.current_time_step]
        self.env.chronics_handler.real_data.curr_iter = self.current_time_step
        new_obs, new_reward, new_done, new_info = self.env.step(self.action)

        self.current_time_step += 1
        # the chronics handler handled the "self.env.chronics_handler.curr_iter += 1"
        return new_obs, new_reward, new_done, new_info

    def go_to(self, time_step):
        """
        goes to the step number "time_step".

        So if you go_to timestep 10 then you retrieve the 10th observation and its as if the
        agent did the 9th action (just before)
        """
        if time_step > len(self.episode_data.actions):
            raise Grid2OpException("The stored episode counts only {} time steps. You cannot go "
                                   "at time step {}"
                                   "".format(len(self.episode_data.actions), time_step))

        if time_step <= 0:
            raise Grid2OpException("You cannot go to timestep <= 0, it does not make sense (as there is not \"-1th\""
                                   "action). If you want to load the data, please use \"EpisodeReboot.load\".")
        self.current_time_step = time_step - 1
        return self.next(update=True)
