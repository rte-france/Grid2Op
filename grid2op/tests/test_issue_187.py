# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings
import numpy as np

import grid2op
from grid2op.dtypes import dt_float
from grid2op.Reward import RedispReward
from grid2op.Runner import Runner


class Issue187Tester(unittest.TestCase):
    """
    this test ensure that every "test" environment can be converted to gym

    this test suit goes beyond the simple error raised in the github issue.
    """

    def setUp(self) -> None:
        self.tol = 1e-5  # otherwise issues with converting to / from float32

    def test_issue_187(self):
        """test the range of the reward class"""
        for env_name in grid2op.list_available_test_env():
            if env_name == "blank":
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                with grid2op.make(
                    env_name, test=True, reward_class=RedispReward,
                    _add_to_name=type(self).__name__
                ) as env:
                    obs = env.reset()
                    obs, reward, done, info = env.step(env.action_space())
                    assert (
                        reward <= env.reward_range[1]
                    ), f"error for reward_max for {env_name}"
                    assert (
                        reward >= env.reward_range[0]
                    ), f"error for reward_min for {env_name}"

    def test_custom_reward(self):
        """test i can generate the reward and use it in the envs"""
        reward_cls = RedispReward.generate_class_custom_params(
            alpha_redisph=2,
            min_load_ratio=0.15,
            worst_losses_ratio=0.05,
            min_reward=-10.0,
            reward_illegal_ambiguous=0.0,
            least_losses_ratio=0.015,
        )

        for env_name in grid2op.list_available_test_env():
            if env_name == "blank":
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                with grid2op.make(env_name, test=True, reward_class=reward_cls, _add_to_name=type(self).__name__) as env:
                    obs = env.reset()
                    obs, reward, done, info = env.step(env.action_space())
                    # test that reward is in the correct range
                    assert (
                        reward <= env.reward_range[1]
                    ), f"error reward > reward_max for {env_name}"
                    assert (
                        reward >= env.reward_range[0]
                    ), f"error reward < reward_min for {env_name}"

                    # test the parameters are effectively changed

                    # what should be computed
                    _alpha_redisph = dt_float(2)
                    _min_load_ratio = dt_float(0.15)
                    _worst_losses_ratio = dt_float(0.05)
                    _min_reward = dt_float(-10.0)
                    _reward_illegal_ambiguous = dt_float(0.0)
                    _least_losses_ratio = dt_float(0.015)

                    worst_marginal_cost = np.max(env.gen_cost_per_MW)
                    worst_load = dt_float(np.sum(env.gen_pmax))
                    # it's not the worst, but definitely an upper bound
                    worst_losses = dt_float(_worst_losses_ratio) * worst_load
                    worst_redisp = _alpha_redisph * np.sum(
                        env.gen_pmax
                    )  # not realistic, but an upper bound
                    max_regret = (worst_losses + worst_redisp) * worst_marginal_cost / 12.
                    reward_min = dt_float(_min_reward)

                    least_loads = dt_float(
                        worst_load * _min_load_ratio
                    )  # half the capacity of the grid
                    least_losses = dt_float(
                        _least_losses_ratio * least_loads
                    )  # 1.5% of losses
                    least_redisp = dt_float(0.0)  # lower_bound is 0
                    base_marginal_cost = np.min(
                        env.gen_cost_per_MW[env.gen_cost_per_MW > 0.0]
                    )
                    min_regret = (least_losses + least_redisp) * base_marginal_cost / 12.
                    reward_max = dt_float((max_regret - min_regret) / least_loads)
                    assert (
                        abs(env.reward_range[1] - reward_max) <= self.tol
                    ), f"wrong reward max computed for {env_name}: {env.reward_range[1]} vs {reward_max}"
                    assert (
                        abs(env.reward_range[0] - reward_min) <= self.tol
                    ), f"wrong reward min computed for {env_name}: {env.reward_range[0]} vs {reward_min}"

    def test_custom_reward_runner(self):
        """test i can generate the reward and use it in the envs"""
        reward_cls = RedispReward.generate_class_custom_params(
            alpha_redisph=2,
            min_load_ratio=0.15,
            worst_losses_ratio=0.05,
            min_reward=-10.0,
            reward_illegal_ambiguous=0.0,
            least_losses_ratio=0.015,
        )
        env_name = "l2rpn_case14_sandbox"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make(env_name, test=True, reward_class=reward_cls, _add_to_name=type(self).__name__) as env:
                obs = env.reset()
                runner = Runner(**env.get_params_for_runner())
                res = runner.run(nb_episode=2, nb_process=2)


if __name__ == "__main__":
    unittest.main()
