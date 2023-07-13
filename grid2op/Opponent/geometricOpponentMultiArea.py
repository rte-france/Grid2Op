#Copyright (c) 2019-2021, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from typing import Optional, List
import numpy as np

from grid2op.dtypes import dt_int

from grid2op.Opponent.baseOpponent import BaseOpponent
from grid2op.Opponent.geometricOpponent import GeometricOpponent
from grid2op.Exceptions import OpponentError


class GeometricOpponentMultiArea(BaseOpponent):
    """
    This opponent is a combination of several similar opponents (of Kind Geometric Opponent at this stage) attacking on different areas.
    The difference between unitary opponents is mainly the attackable lines (which belongs to different pre-identified areas
    """

    def __init__(self, action_space):
        BaseOpponent.__init__(self, action_space)
        self.list_opponents : Optional[List[GeometricOpponent]] = None
        self._new_attack_time_counters : Optional[np.ndarray] = None
        self._previous_attacks = None

    def init(
        self,
        partial_env,
        lines_attacked=None,
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
            The lists of lines attacked by each unitary opponent (this is a list of list: the size 
            of the outer list is the number of underlying opponent / number of areas and for each inner 
            list it gives the name of the lines to attack.)

        attack_every_xxx_hour: ``float``
            see the GeometricOpponent::init documentation

        average_attack_duration_hour: ``float``
            see the GeometricOpponent::init documentation

        minimum_attack_duration_hour: ``int``
            see the GeometricOpponent::init documentation

        pmax_pmin_ratio: ``float``
            see the GeometricOpponent::init documentation

        """

        if lines_attacked is None:
            partial_env.logger.warning("GeometricOpponentMultiArea: no area provided, the opponent will be deactivated.")
            return
        
        self.list_opponents = [GeometricOpponent(action_space=self.action_space) for _ in lines_attacked]
        self._previous_attacks = [None for _ in lines_attacked]

        for lines_attacked, opp in zip(lines_attacked, self.list_opponents):
            opp.init(
                partial_env=partial_env,
                lines_attacked=lines_attacked,
                attack_every_xxx_hour=attack_every_xxx_hour,
                average_attack_duration_hour=average_attack_duration_hour,
                minimum_attack_duration_hour=minimum_attack_duration_hour,
                pmax_pmin_ratio=pmax_pmin_ratio,
                **kwargs,
            )
        self._new_attack_time_counters = np.array([-1 for _ in lines_attacked])#ou plutôt 0 comme dans Geometric Opponent ?

    def reset(self, initial_budget):
        self._new_attack_time_counters = np.array([-1 for _ in self.list_opponents])

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
        self._new_attack_time_counters -= 1
        self._new_attack_time_counters[self._new_attack_time_counters < -1] = -1

        attack_combined = None
        for opp_id, opp in enumerate(self.list_opponents):
            if self._new_attack_time_counters[opp_id] == -1:
                attack_opp, attack_duration_opp = opp.attack(observation, agent_action, env_action, budget, previous_fails)
                if attack_opp is not None:
                    self._new_attack_time_counters[opp_id] = attack_duration_opp
                    self._previous_attacks[opp_id] = attack_opp
                    if attack_combined is None:
                        attack_combined = attack_opp.copy()
                    else:
                        attack_combined += attack_opp
                else:
                    self._previous_attacks[opp_id] = None
            else:
                opp.tell_attack_continues(observation, agent_action, env_action, budget)
                if attack_combined is None:
                    attack_combined = self._previous_attacks[opp_id].copy()
                else:
                    attack_combined += self._previous_attacks[opp_id]
        return attack_combined, 1


    def tell_attack_continues(self, observation, agent_action, env_action, budget):
        raise RuntimeError("I should not get there !")

    def get_state(self):
        return (self._new_attack_time_counters,
                self._previous_attacks,
                [opp.get_state() for opp in self.list_opponents])

    def set_state(self, my_state):
        self._new_attack_time_counters = np.array(my_state[0])
        self._previous_attacks = [el.copy() if el is not None else None for el in my_state[1]]
        for el, opp in zip(my_state[2], self.list_opponents):
            opp.set_state(el)
            
    def _custom_deepcopy_for_copy(self, new_obj, dict_=None):
        new_obj._new_attack_time_counters = 1 * self._new_attack_time_counters
        new_obj._previous_attacks = [el.copy() if el is not None else None 
                                     for el in self._previous_attacks]
        new_obj.list_opponents = []
        for opp in self.list_opponents:
            new_opp = type(opp).__new__(type(opp))
            opp._custom_deepcopy_for_copy(new_opp, dict_)
            new_obj.list_opponents.append(new_opp)
        super()._custom_deepcopy_for_copy(new_obj)
        return new_obj

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
        seeds = []
        super().seed(seed)
        max_seed = np.iinfo(dt_int).max  # 2**32 - 1
        for opp in self.list_opponents:
            this_seed = self.space_prng.randint(max_seed)
            seeds.append(opp.seed(this_seed))
        return (seed, seeds)
