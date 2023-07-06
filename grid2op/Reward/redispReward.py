# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import sys
import numpy as np

import re
from grid2op._glop_platform_info import _IS_WINDOWS, _IS_LINUX, _IS_MACOS
from grid2op.Exceptions import Grid2OpException
from grid2op.Reward.baseReward import BaseReward
from grid2op.dtypes import dt_float


class RedispReward(BaseReward):
    """
    This reward can be used for environments where redispatching is available. It assigns a cost to redispatching action
    and penalizes with the losses.

    This is the closest reward to the score used for the l2RPN competitions.

    Examples
    ---------
    You can use this reward in any environment with:

    .. code-block:

        import grid2op
        from grid2op.Reward import RedispReward

        # then you create your environment with it:
        NAME_OF_THE_ENVIRONMENT = "rte_case14_realistic"
        env = grid2op.make(NAME_OF_THE_ENVIRONMENT,reward_class=RedispReward)
        # and do a step with a "do nothing" action
        obs = env.reset()
        obs, reward, done, info = env.step(env.action_space())
        # the reward is computed with the RedispReward class

        # NB this is the default reward of many environments in the grid2op framework

    This class depends on some "meta parameters". These meta parameters can be changed when the class is created
    in the following way:

    .. code-block:: python

        import grid2op
        from grid2op.Reward import RedispReward

        reward_cls = RedispReward.generate_class_custom_params(alpha_redisph=5,
                                                               min_load_ratio=0.1,
                                                               worst_losses_ratio=0.05,
                                                               min_reward=-10.,
                                                               reward_illegal_ambiguous=0.,
                                                               least_losses_ratio=0.015)
        env_name = "l2rpn_case14_sandbox"  # or any other name
        env = grid2op.make(env_name,reward_class=reward_cls)

    These meta parameters means:

    - alpha_redisp: extra cost paid when performing redispatching. For 1MW of redispatching done, you pay
      "alpha_redisph"
    - min_load_ratio: how to compute the minimum load on the grid, based on the total generation (sum of gen_pmax)
    - worst_losses_ratio: worst loss possible on the grid (5% is an upper bound for normal grid)
    - min_reward: what is the minimum reward of this class (can be parametrized, and is only used when there is
      a game over
    - reward_illegal_ambiguous: reward given when the action is illegal or ambiguous
    - least_losses_ratio: the minimum loss you can have (1.5% of the total demand should be a lower bound for real grid)

    Notes
    ------
    On windows and MacOs, due to a compatibility issue with multi-processing, it is not possible to have different
    "RedisReward" with different meta parameters (see the "Examples" section).

    """

    _alpha_redisp = dt_float(5.0)
    _min_load_ratio = dt_float(0.1)  # min load = min_load_ratio * max_load
    _worst_losses_ratio = dt_float(
        0.05
    )  # worst_losses = worst_losses_ratio * worst_load
    _min_reward = dt_float(-10.0)  # reward when game over
    _reward_illegal_ambiguous = dt_float(
        0.0
    )  # reward when action is illegal or ambiguous
    _least_losses_ratio = dt_float(
        0.015
    )  # least_losses = least_losses_ratio * least_loads

    def __init__(self, logger=None):
        BaseReward.__init__(self, logger=logger)
        self.reward_min = None
        self.reward_max = None
        self.max_regret = dt_float(0.0)
        self.reward_illegal_ambiguous = None

    @classmethod
    def generate_class_custom_params(
        cls,
        alpha_redisph=5.0,
        min_load_ratio=0.1,  # min load = min_load_ratio * max_load
        worst_losses_ratio=0.05,  # worst_losses = worst_losses_ratio * worst_load
        min_reward=-10.0,
        least_losses_ratio=0.015,  # least_losses = least_losses_ratio * least_loads
        reward_illegal_ambiguous=0.0,
    ):
        if _IS_LINUX:
            # on linux it's fine, i can create new classes for each meta parameters
            nm_res = f"RedispReward_{alpha_redisph:.2f}_{min_load_ratio:.2f}_{worst_losses_ratio:.2f}"
            nm_res += f"_{min_reward:.2f}_{least_losses_ratio:.2f}_{reward_illegal_ambiguous:.2f}"
            nm_res = re.sub("\\.", "@", nm_res)
            cls_attr_as_dict = {
                "_alpha_redisp": dt_float(alpha_redisph),
                "_min_load_ratio": dt_float(min_load_ratio),
                "_worst_losses_ratio": dt_float(worst_losses_ratio),
                "_min_reward": dt_float(min_reward),
                "_least_losses_ratio": dt_float(least_losses_ratio),
                "_reward_illegal_ambiguous": dt_float(reward_illegal_ambiguous),
            }
            res_cls = type(nm_res, (cls,), cls_attr_as_dict)
            res_cls.__module__ = cls.__module__
            setattr(sys.modules[cls.__module__], nm_res, res_cls)
            globals()[nm_res] = res_cls
        else:
            # i mess with the default parameters in the base class, i know i know it's not pretty, but hey...

            # TODO make that prettier and clean the way to make the reward in the env (for example allow to pass
            # objects and not just class)
            cls._alpha_redisp = dt_float(alpha_redisph)
            cls._min_load_ratio = dt_float(min_load_ratio)
            cls._worst_losses_ratio = dt_float(worst_losses_ratio)
            cls._min_reward = dt_float(min_reward)
            cls._least_losses_ratio = dt_float(least_losses_ratio)
            cls._reward_illegal_ambiguous = dt_float(reward_illegal_ambiguous)
            res_cls = cls

        return res_cls

    def initialize(self, env):
        if not env.redispatching_unit_commitment_availble:
            raise Grid2OpException(
                "Impossible to use the RedispReward reward with an environment without generators "
                "cost. Please make sure env.redispatching_unit_commitment_availble is available."
            )
        cls_ = type(self)

        worst_marginal_cost = np.max(env.gen_cost_per_MW)
        worst_load = dt_float(np.sum(env.gen_pmax))
        # it's not the worst, but definitely an upper bound
        worst_losses = dt_float(cls_._worst_losses_ratio) * worst_load
        worst_redisp = cls_._alpha_redisp * np.sum(env.gen_pmax)  # not realistic, but an upper bound
        self.max_regret = (worst_losses + worst_redisp) * worst_marginal_cost * env.delta_time_seconds / 3600.0
        self.reward_min = dt_float(cls_._min_reward)

        least_loads = dt_float(
            worst_load * cls_._min_load_ratio
        )  # half the capacity of the grid
        least_losses = dt_float(
            cls_._least_losses_ratio * least_loads * env.delta_time_seconds / 3600.0
        )  # 1.5% of losses
        least_redisp = dt_float(0.0)  # lower_bound is 0
        base_marginal_cost = np.min(env.gen_cost_per_MW[env.gen_cost_per_MW > 0.0])
        min_regret = (least_losses + least_redisp) * base_marginal_cost
        self.reward_max = dt_float((self.max_regret - min_regret) / least_loads)
        self.reward_illegal_ambiguous = cls_._reward_illegal_ambiguous

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        res = None
        if is_done:
            # if the episode is over and it's my fault (i did a blackout) i strongly
            if has_error or is_illegal or is_ambiguous:
                res = self.reward_min
        elif is_illegal or is_ambiguous:
            res = self._reward_illegal_ambiguous

        if res is None:
            # compute the losses
            gen_p, *_ = env.backend.generators_info()
            load_p, *_ = env.backend.loads_info()
            # don't forget to convert MW to MWh !
            losses = (np.sum(gen_p) - np.sum(load_p)) * env.delta_time_seconds / 3600.0

            # compute the marginal cost
            gen_activeprod_t = env._gen_activeprod_t
            marginal_cost = np.max(env.gen_cost_per_MW[gen_activeprod_t > 0.0])

            # redispatching amount
            actual_dispatch = env._actual_dispatch
            redisp_cost = (
                self._alpha_redisp * np.sum(np.abs(actual_dispatch)) * marginal_cost * env.delta_time_seconds / 3600.0
            )

            # cost of losses
            losses_cost = losses * marginal_cost

            # cost of storage
            c_storage = np.sum(np.abs(env._storage_power)) * marginal_cost * env.delta_time_seconds / 3600.0
            
            # total "regret"
            regret = losses_cost + redisp_cost + c_storage

            # compute reward
            reward = self.max_regret - regret

            # divide it by load, to be less sensitive to load variation
            res = dt_float(reward / np.sum(load_p))

        return res
