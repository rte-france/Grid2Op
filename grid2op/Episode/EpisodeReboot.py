# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import copy
import numpy as np

from datetime import timedelta

from grid2op.Exceptions import Grid2OpException
from grid2op.Chronics import GridValue, ChronicsHandler
from grid2op.Opponent import BaseOpponent
from grid2op.Environment import Environment

from grid2op.Episode.EpisodeData import EpisodeData


class _GridFromLog(GridValue):
    """
    /!\ Internal, do not use /!\

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

    def initialize(self, order_backend_loads, order_backend_prods, order_backend_lines, order_backend_subs,
                   names_chronics_to_backend):
        pass

    def load_next(self):
        obs = next(self.episode_data.observations)
        self.current_datetime = obs.get_time_stamp()
        self.curr_iter += 1

        res = {}
        injs = {"prod_p": obs.prod_p,
                "load_p": obs.load_p,
                "load_q": obs.load_q,
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
    pass


class EpisodeReboot:
    def __init__(self):
        self.episode_data = None
        self.env = None
        self.chronics_handler = None

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

        self.env = Environment(**env_kwargs,
                               backend=backend,
                               chronics_handler=self.chronics_handler,
                               opponent_class=OpponentFromLog)

    def go_to(self, time_step):
        if time_step > len(self.episode_data.actions):
            raise Grid2OpException("The stored episode counts only {} time steps. You cannot go "
                                   "at time step {}"
                                   "".format(len(self.episode_data.actions), time_step))
        if time_step == 0:
            return

        # get the state just before
        self.episode_data.go_to(time_step-2)

        # now set the environment state to this value
        obs = next(self.episode_data.observations)
        act = next(self.episode_data.actions)
        self.env._gen_activeprod_t[:] = obs.prod_p
        self.env._actual_dispatch[:] = obs.actual_dispatch
        self.env._target_dispatch[:] = obs.target_dispatch
        self.env._gen_activeprod_t_redisp[:] = obs.prod_p + obs.actual_dispatch
        self.env.current_obs = obs
        self.env._timestep_overflow[:] = obs.timestep_overflow
        self.env._times_before_line_status_actionable[:] = obs.time_before_cooldown_line
        self.env._times_before_topology_actionable[:] = obs.time_before_cooldown_sub

        self.env._duration_next_maintenance[:] = obs.duration_next_maintenance
        self.env._time_next_maintenance[:] = obs.time_next_maintenance

        # TODO check that the "stored" "last bus for when the powerline were connected" are
        # kept there (I might need to do a for loop)
        # to test that i might need to use a "change status" and see if powerlines are connected
        # to the right bus
        self.env._backend_action += self.env.action_space({"set_bus": obs.topo_vect})
        self.env.backend.apply_action(self.env._backend_action)
        disc_lines, detailed_info, conv_ = self.env.backend.next_grid_state(env=self.env)
        self.env._backend_action.update_state(disc_lines)
        self.env._backend_action.reset()
        obs2, reward2, done2, info2 = self.env.step(act)
        import pdb
        pdb.set_trace()
