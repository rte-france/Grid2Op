# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import numpy as np
from grid2op.Agent.recoPowerlineAgent import RecoPowerlineAgent
import pandas as pd

#try:
#    from lightsim2grid import LightSimBackend
#
#    bkclass = LightSimBackend
#    # raise ImportError()
#except ImportError as excq_:
#    from grid2op.Backend import PandaPowerBackend



class AlertAgent(RecoPowerlineAgent):
    """
    This is a :class:`AlertAgent` example, which will attempt to reconnect powerlines and send alerts on the worst possible attacks: for each disconnected powerline
    that can be reconnected, it will simulate the effect of reconnecting it. And reconnect the one that lead to the
    highest simulated reward. It will also simulate the effect of having a line disconnection on attackable lines and raise alerts for the worst ones

    """

    def __init__(self, action_space,percentage_alert=30,simu_step=0):#[0, 9, 13, 14, 18, 23, 27, 39, 45, 56]):
        RecoPowerlineAgent.__init__(self, action_space)
        self.percentage_alert = percentage_alert
        #self.alertable_line_ids=alertable_line_ids
        self.simu_step=simu_step

    def act(self, observation, reward, done=False):
        action=super().act(observation, reward, done=False)
        alertable_line_ids=observation.alertable_line_ids
        #if (self.alertable_line_ids is None):
        #    self.alertable_line_ids = [i for i in range(observation.n_line) if observation.name_line[i] in observation.alertable_line_names]

        #simu d'analyse de sécurité à chaque pas de temps sur les lignes attaquées
        n_alertable_lines=len(alertable_line_ids)
        nb_overloads=np.zeros(n_alertable_lines)
        rho_max_N_1=np.zeros(n_alertable_lines)

        # test which backend to know which method to call

        N_1_actions=[self.action_space({"set_line_status": [(id_, -1)]}) for id_ in alertable_line_ids]
        for i, action in enumerate(N_1_actions):

            #check that line is not already disconnected
            if (observation.line_status[alertable_line_ids[i]]):
                (
                    simul_obs,
                    simul_reward,
                    simul_has_error,
                    simul_info,
                ) = observation.simulate(action,time_step=self.simu_step)

                rho_simu=simul_obs.rho
                if(not simul_has_error):
                    nb_overloads[i]=np.sum(rho_simu >= 1)
                    rho_max_N_1[i]=np.max(rho_simu)

        df_to_sort=pd.DataFrame({"nb_overloads":nb_overloads,"rho_max_N_1":rho_max_N_1})
        indices=df_to_sort.sort_values(['nb_overloads','rho_max_N_1'], ascending=False).index

        #alerts to send
        indices_to_keep=list(indices[0:int(self.percentage_alert/100*n_alertable_lines)])
        action.raise_alert = [i for i in indices_to_keep]#[self.alertable_line_ids[i] for i in indices_to_keep]#[attackable_line_id]

        return action
