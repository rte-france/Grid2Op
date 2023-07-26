# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
from grid2op.Action import BaseAction
from grid2op.Agent.recoPowerlineAgent import RecoPowerlineAgent
from grid2op.Agent.baseAgent import BaseAgent
from grid2op.Observation import BaseObservation
from grid2op.dtypes import dt_int


class AlertAgent(BaseAgent):
    """
    This is a :class:`AlertAgent` example, which will attempt to reconnect powerlines and send alerts on the worst possible attacks: for each disconnected powerline
    that can be reconnected, it will simulate the effect of reconnecting it. And reconnect the one that lead to the
    highest simulated reward. It will also simulate the effect of having a line disconnection on attackable lines and raise alerts for the worst ones

    """

    def __init__(self,
                 action_space,
                 grid_controler=RecoPowerlineAgent,
                 percentage_alert=30,
                 simu_step=1,
                 threshold=0.99):
        super().__init__(action_space)
        if isinstance(grid_controler, type):
            self.grid_controler = grid_controler(action_space)
        else:
            self.grid_controler = grid_controler
            
        self.percentage_alert = percentage_alert
        self.simu_step = simu_step
        self.threshold = threshold  # if the max flow after a line disconnection is below threshold, then the alert is not raised
        
        # store the result of the simulation of powerline disconnection
        self.alertable_line_ids = type(action_space).alertable_line_ids
        self.n_alertable_lines = len(self.alertable_line_ids)
        self.nb_overloads = np.zeros(self.n_alertable_lines, dtype=dt_int)
        self.rho_max_N_1 = np.zeros(self.n_alertable_lines)
        self.N_1_actions = [self.action_space({"set_line_status": [(id_, -1)]}) for id_ in self.alertable_line_ids]
        self._first_k = np.zeros(self.n_alertable_lines, dtype=bool)
        self._first_k[:int(self.percentage_alert / 100. * self.n_alertable_lines)] = True
        
    def act(self, observation: BaseObservation, reward: float, done: bool = False) -> BaseAction:
        action = self.grid_controler.act(observation, reward, done)

        self.nb_overloads[:] = 0
        self.rho_max_N_1[:] = 0.
        
        # test which backend to know which method to call
        for i, tmp_act in enumerate(self.N_1_actions):
            # only simulate if the line is connected
            if observation.line_status[self.alertable_line_ids[i]]:
                action_to_simulate = tmp_act
                action_to_simulate += action
                action_to_simulate.remove_line_status_from_topo(observation)
                (
                    simul_obs,
                    simul_reward,
                    simul_done,
                    simul_info,
                ) = observation.simulate(action_to_simulate, time_step=self.simu_step)

                rho_simu = simul_obs.rho
                if not simul_done:
                    self.nb_overloads[i] = (rho_simu >= 1).sum()
                    self.rho_max_N_1[i] = (rho_simu).max()
                else:
                    self.nb_overloads[i] = type(observation).n_line
                    self.rho_max_N_1[i] = 5.
        
        # sort the index by nb_overloads and, if nb_overloads is equal, sort by rho_max
        ind = (self.nb_overloads * 1000. + self.rho_max_N_1).argsort()
        ind = ind[::-1]
        
        # send alerts when the powerline is among the top k (not to send too many alerts) and 
        # the max rho after the powerline disconnection is too high (above threshold)
        indices_to_keep = ind[self._first_k & (self.rho_max_N_1[ind] >= self.threshold)]
        action.raise_alert = [i for i in indices_to_keep]

        return action
