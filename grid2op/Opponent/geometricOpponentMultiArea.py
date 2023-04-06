#Copyright (c) 2019-2021, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import copy
import numpy as np
from grid2op.Exceptions.Grid2OpException import Grid2OpException

from grid2op.dtypes import dt_int
import os
import json
from grid2op.Opponent import BaseOpponent
from grid2op.Opponent import GeometricOpponent
from grid2op.Exceptions import OpponentError


class GeometricOpponentMultiArea(BaseOpponent):
    """
    This opponent is a combination of several similar opponents (of Kind Geometric Opponent at this stage) attacking on different areas.
    The difference between unitary opponents is mainly the attackable lines (which belongs to different pre-identified areas
    """

    GRID_AREA_FILE_NAME = "grid_areas.json"

    def __init__(self, action_space):
        BaseOpponent.__init__(self, action_space)
        self.list_opponents=[]
        self._new_attack_time_counters=[]
        self._is_opp_attack_continue=[]

    def init(
        self,
        partial_env,
        lines_attacked=None,#list(list()),
        attack_every_xxx_hour=24,
        average_attack_duration_hour=4,
        minimum_attack_duration_hour=2,
        pmax_pmin_ratio=4,
        **kwargs,
    ):
        """
        Generic function used to initialize the derived classes. For example, if an opponent reads from a file, the
        path where is the file is located should be pass with this method.
        This is based on init from GeometricOpponent, only parameter lines_attacked becomes a list of list

        Parameters
        ----------
        partial_env: grid2op Environment
            see the GeometricOpponent::init documentation

        lines_attacked: ``list(list)``
            The lists of lines attacked by each unitary opponent

        attack_every_xxx_hour: ``float``
            see the GeometricOpponent::init documentation

        average_attack_duration_hour: ``float``
            see the GeometricOpponent::init documentation

        minimum_attack_duration_hour: ``int``
            see the GeometricOpponent::init documentation

        pmax_pmin_ratio: ``float``
            see the GeometricOpponent::init documentation

        """

        self.list_opponents=[GeometricOpponent(action_space=partial_env.action_space) for el in lines_attacked]

        for el,opp in zip(lines_attacked,self.list_opponents):
            opp.init(
                partial_env=partial_env,
                lines_attacked=el,
                attack_every_xxx_hour=attack_every_xxx_hour,
                average_attack_duration_hour=average_attack_duration_hour,
                minimum_attack_duration_hour=minimum_attack_duration_hour,
                pmax_pmin_ratio=pmax_pmin_ratio,
                **kwargs,
            )
        self._new_attack_time_counters=np.array([-1 for el in lines_attacked])#ou plutôt 0 comme dans Geometric Opponent ?
        self._is_opp_attack_continue=np.array([False for el in lines_attacked])


    def reset(self, initial_budget):
        for opp in self.list_opponents:  # maybe loop in different orders each time
            opp.reset(initial_budget)



    def attack(self, observation, agent_action, env_action, budget, previous_fails):
        """
        This method is the equivalent of "attack" for a regular agent.
        Opponent, in this framework can have more information than a regular agent (in particular it can
        view time step t+1), it has access to its current budget etc.
        Here we take the combination of unitary opponent attacks if they happen at the same time.
        We choose the attack duration as the minimum duration of several simultaneous attacks if that happen.

        Parameters
        ----------
        observation: :class:`grid2op.Observation.Observation`
            see the GeometricOpponent::attack documentation
        opp_reward: ``float``
            see the GeometricOpponent::attack documentation
        done: ``bool``
            see the GeometricOpponent::attack documentation
        agent_action: :class:`grid2op.Action.Action`
            see the GeometricOpponent::attack documentation
        env_action: :class:`grid2op.Action.Action`
            see the GeometricOpponent::attack documentation
        budget: ``float``
            see the GeometricOpponent::attack documentation
        previous_fails: ``bool``
            see the GeometricOpponent::attack documentation
        Returns
        -------
        attack: :class:`grid2op.Action.Action`
            see the GeometricOpponent::attack documentation
        duration: ``int``
            see the GeometricOpponent::attack documentation
        """

        #go through opponents and check if attack or not. As soon as one attack, stop there
        self._new_attack_time_counters+=-1
        self._new_attack_time_counters[self._new_attack_time_counters<-1]=-1

        attack_combined=None
        attack_combined_duration=None

        for i,opp in enumerate(self.list_opponents):#maybe loop in different orders each time
            opp._new_attack_time_counters[i]

            if(self._new_attack_time_counters[i]==-1):
                attack_opp, attack_duration_opp=opp.attack(observation, agent_action, env_action, budget, previous_fails)

                if(attack_opp is not None):
                    self._new_attack_time_counters[i]=attack_duration_opp
                    self._is_opp_attack_continue[i]=True
                    if attack_combined is None:
                        attack_combined=attack_opp
                        attack_combined_duration=attack_duration_opp
                    else:
                        attack_combined+=attack_opp
                        if attack_duration_opp<attack_combined_duration:
                            attack_combined_duration=attack_duration_opp
            else:
                opp.tell_attack_continues()

        return attack_combined,attack_combined_duration


    def tell_attack_continues(self, observation, agent_action, env_action, budget):
        pass

    def get_state(self):#ou get_state avec nom de l'opponent ?
        return (
            [opp.get_state() for opp in self.list_opponents],
        )

    def set_state(self, my_state):#ou set_state avec nom de l'opponent ?
        #check that the dimensions of each array are in the number of opponents ?

        for el,opp in zip(my_state,self.list_opponents):
            opp.set_state(el)



    def seed(self, seed):
        """
        INTERNAL

         .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            We do not recommend to use this function outside of the two examples given in the description of this class.

        Set the seeds of the source of pseudo random number used for these several unitary opponents.

        Parameters
        ----------
        seed: ``int``
            The root seed to be set for the random number generator.

        Returns
        -------
        seeds: ``list``
            The associated list of seeds used.

        """

        max_seed = np.iinfo(dt_int).max  # 2**32 - 1
        seeds=[]
        super().seed(seed)
        for opp in self.list_opponents:
            seed = self.space_prng.randint(max_seed)

            seeds.append(opp.seed(seed))
        return seeds

