# Copyright (c) 2019-2021, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import copy
import numpy as np

from grid2op.dtypes import dt_int
from grid2op.Opponent import BaseOpponent
from grid2op.Exceptions import OpponentError


class GeometricOpponent(BaseOpponent):
    """
    This opponent will disconnect lines randomly among the attackable lines `lines_attacked`.
    The sampling is done according to the lines load factor (ratio <current going through the line> to <thermal limit
    of the line>)
    (see init for more details).

    The time of the attack is sampled according to a geometric distribution
    """

    def __init__(self, action_space):
        BaseOpponent.__init__(self, action_space)
        self._do_nothing = None
        self._attacks = None
        self._lines_ids = None
        self._next_attack_time = None
        self._attack_hazard_rate = None
        self._recovery_minimum_duration = None
        self._recovery_rate = None
        self._pmax_pmin_ratio = None
        self._attack_times = None
        self._attack_waiting_times = None
        self._attack_durations = None
        self._attack_counter = None
        self._number_of_attacks = None
        self._episode_max_time = None
        self._env = None  # I need to keep a pointer to the environment for computing the maximum length of the episode
        # this is the constructor:
        # it should have the exact same signature as here

    def init(
        self,
        partial_env,
        lines_attacked=(),
        attack_every_xxx_hour=24,
        average_attack_duration_hour=4,
        minimum_attack_duration_hour=2,
        pmax_pmin_ratio=4,
        **kwargs,
    ):
        """
        Generic function used to initialize the derived classes. For example, if an opponent reads from a file, the
        path where is the file is located should be pass with this method.

        Parameters
        ----------
        partial_env: grid2op Environment
            A pointer to the environment that initializes the opponent

        lines_attacked: ``list``
            The list of lines that the XPOpponent should be able to disconnect

        attack_every_xxx_hour: ``float``
            Provide the average duration between two attacks. Note that this should be greater
            than `average_attack_duration_hour` as, for now, an agent can only do one consecutive attack.
            You should provide it in "number of hours" and not in "number of steps"

            It is used to compute the `attack_hazard_rate`.
            Attacks time are sampled with a duration distribution. For this opponent, we use the simplest of these
            distributions : The geometric disribution
            https://en.wikipedia.org/wiki/Geometric_distribution (the discrete time counterpart of the exponential
            distribution).
            The attack_hazard_rate is the main parameter of this distribution. It can be seen as the (constant)
            probability of having an attack
            in the next step. It is also the inverse of the expectation of the time to an attack.

        average_attack_duration_hour: ``float``
            Give, in number of hours, the average attack duration. This should be greater than
            `recovery_minimum_duration_hour`

            Used to compute the `recovery_rate`:
            Recovery times are random or at least should have a random part.
            In our case, we will say that the recovery time is equal to a fixed time (safety procedure time) plus a
            random time (investigations
            and repair operations) sampled according to a geometric distribution

        minimum_attack_duration_hour: ``int``
            Minimum duration of an attack (give it in hour)

        pmax_pmin_ratio: ``float``
            Ratio between the probability of the most likely line to be disconnected and the least likely one.
        """
        self._env = partial_env

        if len(lines_attacked) == 0:
            warnings.warn(
                f"The opponent is deactivated as there is no information as to which line to attack. "
                f'You can set the argument "kwargs_opponent" to the list of the line names you want '
                f' the opponent to attack in the "make" function.'
            )

        # Store attackable lines IDs
        self._lines_ids = []
        for l_name in lines_attacked:
            l_id = np.where(self.action_space.name_line == l_name)
            if len(l_id) and len(l_id[0]):
                self._lines_ids.append(l_id[0][0])
            else:
                raise OpponentError(
                    'Unable to find the powerline named "{}" on the grid. For '
                    "information, powerlines on the grid are : {}"
                    "".format(l_name, sorted(self.action_space.name_line))
                )

        # Pre-build attacks actions
        self._do_nothing = self.action_space({})
        self._attacks = []
        for l_id in self._lines_ids:
            a = self.action_space({"set_line_status": [(l_id, -1)]})
            self._attacks.append(a)
        self._attacks = np.array(self._attacks)

        # Opponent's attack and recovery rates and minimum duration
        # number of steps per hour
        ts_per_hour = 3600.0 / partial_env.delta_time_seconds
        self._recovery_minimum_duration = int(
            minimum_attack_duration_hour * ts_per_hour
        )
        if average_attack_duration_hour < minimum_attack_duration_hour:
            raise OpponentError(
                "The average duration of an attack cannot be lower than the minimum time of an attack"
            )
        elif average_attack_duration_hour == minimum_attack_duration_hour:
            raise OpponentError(
                "Case average_attack_duration_hour == minimum_attack_duration_hour is not supported "
                "at the moment"
            )
        self._recovery_rate = 1.0 / (
            ts_per_hour * (average_attack_duration_hour - minimum_attack_duration_hour)
        )
        if attack_every_xxx_hour <= average_attack_duration_hour:
            raise OpponentError(
                "attack_every_xxx_hour <= average_attack_duration_hour is not supported at the moment."
            )
        self._attack_hazard_rate = 1.0 / (
            ts_per_hour * (attack_every_xxx_hour - average_attack_duration_hour)
        )

        # Opponent's pmax pmin ratio
        self._pmax_pmin_ratio = pmax_pmin_ratio

        # Episode max time
        self._episode_max_time = self._get_episode_duration()

        # Sample attack times and durations for the whole episode
        self.sample_attack_times_and_durations()

        # Set the attack counter to 0
        self._attack_counter = 0

    def _get_episode_duration(self):
        tmp = self._env.max_episode_duration()
        if (not np.isfinite(tmp)) or (tmp == np.iinfo(tmp).max):
            raise OpponentError(
                "Geometric opponent only works (for now) with a known finite episode duration."
            )
        return tmp

    def reset(self, initial_budget):
        # Sample attack times and durations for the whole episode
        self.sample_attack_times_and_durations()

        # Reset the attack counter to 0
        self._attack_counter = 0

        self._next_attack_time = None

        # Episode max time
        self._episode_max_time = self._get_episode_duration()

    def sample_attack_times_and_durations(self):
        self._attack_times = []
        self._attack_waiting_times = []
        self._attack_durations = []
        self._number_of_attacks = 0

        t = 0  # t=0 at the beginning of the episode

        while t < self._episode_max_time:
            # Sampling the next time to attack
            t_to_attack = self.space_prng.geometric(p=self._attack_hazard_rate)
            t_of_attack = t + t_to_attack
            t = t_of_attack
            if t < self._episode_max_time:
                self._attack_waiting_times.append(t_to_attack)
                self._attack_times.append(t_of_attack)
                self._number_of_attacks += 1
                # Sampling the attack duration
                attack_duration = (
                    self._recovery_minimum_duration
                    + self.space_prng.geometric(p=self._recovery_rate)
                )
                self._attack_durations.append(attack_duration)
                t = t + attack_duration
            # TODO : Log these times and durations in a log.file

        self._attack_times = np.array(self._attack_times).astype(dt_int)
        self._attack_waiting_times = np.array(self._attack_waiting_times).astype(dt_int)
        self._attack_durations = np.array(self._attack_durations).astype(dt_int)

    def tell_attack_continues(self, observation, agent_action, env_action, budget):
        self._next_attack_time = None

    def attack(self, observation, agent_action, env_action, budget, previous_fails):
        """
        This method is the equivalent of "attack" for a regular agent.
        Opponent, in this framework can have more information than a regular agent (in particular it can
        view time step t+1), it has access to its current budget etc.
        Parameters
        ----------
        observation: :class:`grid2op.Observation.Observation`
            The last observation (at time t)
        opp_reward: ``float``
            THe opponent "reward" (equivalent to the agent reward, but for the opponent) TODO do i add it back ???
        done: ``bool``
            Whether the game ended or not TODO do i add it back ???
        agent_action: :class:`grid2op.Action.Action`
            The action that the agent took
        env_action: :class:`grid2op.Action.Action`
            The modification that the environment will take.
        budget: ``float``
            The current remaining budget (if an action is above this budget, it will be replaced by a do nothing.
        previous_fails: ``bool``
            Wheter the previous attack failed (due to budget or ambiguous action)
        Returns
        -------
        attack: :class:`grid2op.Action.Action`
            The attack performed by the opponent. In this case, a do nothing, all the time.
        duration: ``int``
            The duration of the attack (if ``None`` then the attack will be made for the longest allowed time)
        """
        # During creation of the environment, do not attack
        if observation is None:
            return None, None

        # If there are no more attacks to come, do not attack
        if self._attack_counter >= self._number_of_attacks:
            return None, None

        if previous_fails:
            # i cannot do the attack, it failed (so self._attack_counter >= 1)
            self._next_attack_time = (
                self._attack_waiting_times[self._attack_counter]
                + self._attack_durations[self._attack_counter - 1]
            )

        # Set the time of the next attack
        if self._next_attack_time is None:
            self._next_attack_time = (
                1 + self._attack_waiting_times[self._attack_counter]
            )

        attack_duration = self._attack_durations[self._attack_counter]
        self._next_attack_time -= 1

        # If the attack time has not come yet, do not attack
        if self._next_attack_time > 0:
            return None, None
        else:
            # Attack is launched
            self._attack_counter += 1

        # If all attackable lines are disconnected, abort attack
        status = observation.line_status[self._lines_ids]
        if np.all(status == False):
            return None, None

        available_attacks = self._attacks[status]

        # If we have a unique attackable line we just attack it
        if len(available_attacks) == 1:
            return available_attacks[0], attack_duration

        # We have several lines, so we need to choose one
        # This will be according to their load factor (rho)
        rho = observation.rho[self._lines_ids][status]

        # The rho_rank vector is the ranking of the lines according
        # to their rho (load factor)
        # 0 : for the line with the lowest load factor
        # (n_attackable_lines - 1) : for the line with the highest load factor
        temp = rho.argsort()
        rho_ranks = np.empty_like(temp)
        rho_ranks[temp] = np.arange(len(rho))

        # We choose the attacked line using a Boltzmann distribution
        # on rho ranks, with a beta parameter (temperature) set to ensure
        # that the probability ratio between the most and the least prefered
        # lines is equal to the pmax_pmin_ratio parameter
        n_attackable_line = len(available_attacks)
        b_beta = np.log(self._pmax_pmin_ratio) / (n_attackable_line - 1)
        raw_probabilities = np.exp(b_beta * rho_ranks)
        b_probabilities = raw_probabilities / raw_probabilities.sum()
        attack = self.space_prng.choice(available_attacks, p=b_probabilities)
        return attack, attack_duration

    def get_state(self):
        return (
            self._attack_times,
            self._attack_waiting_times,
            self._attack_durations,
            self._number_of_attacks,
        )

    def set_state(self, my_state):
        (
            _attack_times,
            _attack_waiting_times,
            _attack_durations,
            _number_of_attacks,
        ) = my_state
        self._attack_times = 1 * _attack_times
        self._attack_waiting_times = 1 * _attack_waiting_times
        self._attack_durations = 1 * _attack_durations
        self._number_of_attacks = 1 * _number_of_attacks

    def _custom_deepcopy_for_copy(self, new_obj, dict_=None):
        super()._custom_deepcopy_for_copy(new_obj, dict_)
        if dict_ is None:
            raise OpponentError("Impossible to deep copy an Opponent without a pointer "
                                "to the original env, named `partial_env`.")

        new_obj._attacks = copy.deepcopy(self._attacks)
        new_obj._lines_ids = copy.deepcopy(self._lines_ids)
        new_obj._next_attack_time = copy.deepcopy(self._next_attack_time)
        new_obj._attack_hazard_rate = copy.deepcopy(self._attack_hazard_rate)
        new_obj._recovery_minimum_duration = copy.deepcopy(
            self._recovery_minimum_duration
        )
        new_obj._recovery_rate = copy.deepcopy(self._recovery_rate)
        new_obj._pmax_pmin_ratio = copy.deepcopy(self._pmax_pmin_ratio)
        new_obj._attack_counter = copy.deepcopy(self._attack_counter)
        new_obj._episode_max_time = copy.deepcopy(self._episode_max_time)
        new_obj._env = dict_[
            "partial_env"
        ]  # I need to keep a pointer to the environment for computing the maximum length of the episode
